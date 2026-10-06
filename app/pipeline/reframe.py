"""Smart vertical reframing: turns a 16:9 (or any) source into a 9:16 Short by
tracking the speaker's face and panning a crop window, instead of a dumb
center-crop that cuts people's heads off.

Approach (fully local, no cloud vision API):
  1. Sample frames every `sample_interval` seconds with OpenCV.
  2. Run a Haar-cascade face detector (bundled with opencv) on each sample.
  3. Track the primary face's center across samples, holding the last known
     position when detection drops out (someone looks away, motion blur, ...).
  4. Smooth that center path with a moving average so the crop pans instead
     of jittering.
  5. Feed the resulting path into ffmpeg's `crop` filter as a time-based
     step expression, then scale to the target resolution.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from .ffmpeg_util import get_ffmpeg, get_ffprobe

logger = logging.getLogger("shorts_ai.reframe")

_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float


def probe_video(path: str | Path) -> VideoInfo:
    cmd = [
        get_ffprobe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    return VideoInfo(width=int(stream["width"]), height=int(stream["height"]), duration=duration, fps=fps)


def _detect_primary_face_center(frame, axis: str) -> float | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Primary = largest bounding box (closest / main speaker)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x + w / 2) if axis == "x" else (y + h / 2)


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def build_crop_plan(
    clip_path: str | Path,
    target_width: int,
    target_height: int,
    smoothing_window: int = 5,
    sample_interval: float = 0.5,
) -> dict:
    """Returns a dict describing the crop: dimensions, pan axis, and a list of
    (time, position) breakpoints, plus whether any face was ever found."""
    info = probe_video(clip_path)
    target_ratio = target_width / target_height
    src_ratio = info.width / info.height

    if src_ratio > target_ratio:
        # source is wider than target -> crop full height, pan horizontally
        axis = "x"
        crop_h = info.height
        crop_w = int(round(info.height * target_ratio / 2) * 2)
        max_pos = info.width - crop_w
        default_center = info.width / 2
    else:
        # source is taller/narrower than target -> crop full width, pan vertically
        axis = "y"
        crop_w = info.width
        crop_h = int(round(info.width / target_ratio / 2) * 2)
        max_pos = info.height - crop_h
        default_center = info.height / 2

    cap = cv2.VideoCapture(str(clip_path))
    centers: list[float] = []
    times: list[float] = []
    found_any = False

    t = 0.0
    while t < info.duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        center = _detect_primary_face_center(frame, axis)
        if center is not None:
            found_any = True
        elif centers:
            center = centers[-1]  # hold last known position
        else:
            center = default_center
        times.append(t)
        centers.append(center)
        t += sample_interval
    cap.release()

    if not times:
        times, centers = [0.0], [default_center]

    smoothed = _smooth(centers, smoothing_window)
    # convert center-of-face to top-left crop coordinate, clamped in range
    positions = [max(0, min(max_pos, c - (crop_w if axis == "x" else crop_h) / 2)) for c in smoothed]

    return {
        "axis": axis,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "breakpoints": list(zip(times, positions)),
        "found_face": found_any,
        "src": info,
    }


def _decimate_breakpoints(
    breakpoints: list[tuple[float, float]], max_points: int = 80, min_delta: float = 3.0
) -> list[tuple[float, float]]:
    """ffmpeg's filter-expression parser has a hard nesting-depth limit —
    empirically, a crop expression built from >=100 chained if()s fails to
    even parse (exit code 234 / EINVAL) rather than just being slow. At the
    default 0.5s sample interval, any clip longer than ~50 seconds produces
    100+ breakpoints, which silently broke reframing for most real clips
    (the highlight picker's own target range goes up to 59s).

    Fixes it two ways: first collapse points that haven't moved meaningfully
    since the last kept one (a still face doesn't need a new step every
    0.5s), then hard-cap to `max_points` with even spacing regardless of how
    much motion there was — so the expression always stays well under
    ffmpeg's limit no matter the clip length or how much the subject moves.
    """
    if len(breakpoints) <= 1:
        return breakpoints

    merged = [breakpoints[0]]
    for t, pos in breakpoints[1:]:
        if abs(pos - merged[-1][1]) >= min_delta:
            merged.append((t, pos))
    if merged[-1][0] != breakpoints[-1][0]:
        merged.append(breakpoints[-1])  # keep the true end time exact

    if len(merged) > max_points:
        step = (len(merged) - 1) / (max_points - 1)
        merged = [merged[round(i * step)] for i in range(max_points)]

    return merged


def _build_step_expr(breakpoints: list[tuple[float, float]]) -> str:
    """Builds an ffmpeg-eval nested if() expression that LINEARLY INTERPOLATES
    between consecutive breakpoints, rather than snapping straight to each
    one. (Name kept as "_build_step_expr" for now since render.py/tests don't
    reference it directly, but it no longer steps.)

    The original version stepped: it held a constant position and then
    teleported instantly to the next breakpoint's position the moment t
    crossed that breakpoint's time. With breakpoints spaced well under a
    second apart, that produces a visibly jerky, stroboscopic pan -- the crop
    window jumps instead of gliding -- which is exactly what made panning
    look rough. Interpolating linearly between each pair of breakpoints makes
    the crop position change continuously over time, which is what an actual
    camera pan looks like.
    """
    if len(breakpoints) == 1:
        return f"{breakpoints[0][1]:.1f}"

    # Value once t reaches/exceeds the final breakpoint: hold it there.
    expr = f"{breakpoints[-1][1]:.1f}"

    for (t0, p0), (t1, p1) in reversed(list(zip(breakpoints[:-1], breakpoints[1:]))):
        span = t1 - t0
        if span <= 0:
            # Degenerate (duplicate timestamps) -- fall back to a flat step
            # for this segment rather than dividing by zero.
            segment = f"{p0:.1f}"
        else:
            slope = (p1 - p0) / span
            segment = f"({p0:.3f}+{slope:.4f}*(t-{t0:.3f}))"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"

    return expr


def apply_reframe(
    input_path: str | Path,
    output_path: str | Path,
    target_width: int = 1080,
    target_height: int = 1920,
    smoothing_window: int = 5,
) -> dict:
    """Crops+scales `input_path` to a `target_width`x`target_height` vertical
    video that follows the detected speaker, writing to `output_path`. Audio
    is preserved. Returns the crop plan metadata (useful for logging/UI)."""
    plan = build_crop_plan(input_path, target_width, target_height, smoothing_window)
    breakpoints = _decimate_breakpoints(plan["breakpoints"])
    expr = _build_step_expr(breakpoints)

    if plan["axis"] == "x":
        crop = f"crop=w={plan['crop_w']}:h={plan['crop_h']}:x='{expr}':y=0"
    else:
        crop = f"crop=w={plan['crop_w']}:h={plan['crop_h']}:x=0:y='{expr}'"

    vf = f"{crop},scale={target_width}:{target_height},setsar=1"

    cmd = [
        get_ffmpeg(), "-y", "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        str(output_path),
    ]
    logger.info("Reframing %s -> %s (axis=%s, face detected=%s)",
                input_path, output_path, plan["axis"], plan["found_face"])
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return plan
