"""Fast touch-ups for an ALREADY-rendered short: trimming the final file and
re-applying captions with chosen keywords emphasized.

Deliberately separate from render.py's initial pipeline: these operate on
the finished output (or its cached work-dir intermediate, for caption
re-burns) instead of re-running cut -> reframe -> overlay from the original
source, which is what keeps them quick. The tradeoff (accepted explicitly
when this was built): trimming the final file doesn't re-time captions or
overlays to the new boundaries, so a caption or stat-card that was near the
old edge can end abruptly right at the new one. A from-scratch re-render
would fix that but takes as long as the original render did.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .captions import build_ass, burn_subtitles, write_ass_file
from .ffmpeg_util import get_ffmpeg
from .transcribe import Word

logger = logging.getLogger("shorts_ai.postedit")


class PostEditError(RuntimeError):
    pass


def trim_rendered_clip(
    input_path: Path,
    tmp_output_path: Path,
    final_output_path: Path,
    trim_start: float,
    trim_end: float,
    current_duration: float,
) -> float:
    """Cuts `trim_start` seconds off the front and `trim_end` off the back of
    an already-rendered clip. Writes to `tmp_output_path` first and only
    replaces `final_output_path` on success, so a failed ffmpeg run never
    corrupts the existing output. Returns the new duration."""
    trim_start = max(0.0, trim_start)
    trim_end = max(0.0, trim_end)
    new_duration = current_duration - trim_start - trim_end
    if new_duration <= 1.0:
        raise PostEditError(
            f"That trim would leave only {max(new_duration, 0):.1f}s of video -- pick a smaller trim."
        )

    fast_seek = max(0.0, trim_start - 2.0)
    remainder = trim_start - fast_seek
    cmd = [
        get_ffmpeg(), "-y",
        "-ss", f"{fast_seek:.3f}", "-i", str(input_path),
        "-ss", f"{remainder:.3f}", "-t", f"{new_duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(tmp_output_path),
    ]
    logger.info("Trimming %s: -%.1fs start, -%.1fs end -> %.1fs", input_path, trim_start, trim_end, new_duration)
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    tmp_output_path.replace(final_output_path)
    return new_duration


def cut_base_clip(source: Path, out_path: Path, head: float, duration: float) -> None:
    """Writes a copy of the caption-free `source` clip with `head` seconds
    removed from the front and the given total `duration` kept. Used so that
    caption edits after a trim start from the *trimmed* footage (otherwise
    re-burning would silently bring the trimmed-off seconds back, and the
    captions would drift out of sync with the video)."""
    fast_seek = max(0.0, head - 2.0)
    cmd = [
        get_ffmpeg(), "-y",
        "-ss", f"{fast_seek:.3f}", "-i", str(source),
        "-ss", f"{head - fast_seek:.3f}", "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostEditError(f"ffmpeg failed while preparing the clip:\n{proc.stderr[-800:]}")


def remove_captions(base_video: Path, tmp_output_path: Path, final_output_path: Path) -> None:
    """Replaces the final short with the caption-free `base_video` (stream
    copy -- near-instant)."""
    cmd = [get_ffmpeg(), "-y", "-i", str(base_video), "-c", "copy", str(tmp_output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostEditError(f"ffmpeg failed while removing captions:\n{proc.stderr[-800:]}")
    tmp_output_path.replace(final_output_path)


def reburn_captions_with_emphasis(
    source_video_path: Path,
    ass_path: Path,
    tmp_output_path: Path,
    final_output_path: Path,
    words: list[Word],
    clip_start: float,
    emphasis_words: set[str],
    video_width: int,
    video_height: int,
    caption_cfg,
) -> None:
    """Rebuilds the .ass subtitle track with the given emphasis words and
    re-burns it onto `source_video_path` (the clip's pre-caption
    intermediate from the original render, e.g. work/clipN_02_overlaid.mp4),
    replacing `final_output_path` on success. Needs the clip's own word list
    (sliced from the cached transcript) and the caption style config the
    clip was originally rendered with (font/size/colors/position/...)."""
    if not source_video_path.exists():
        raise PostEditError(
            "The pre-caption version of this clip is no longer on disk (its job folder may have "
            "been cleaned up), so captions can't be re-applied without a full re-render."
        )

    ass_content = build_ass(
        words, clip_start,
        video_width=video_width, video_height=video_height,
        font=caption_cfg.font, font_size=caption_cfg.font_size,
        highlight_color=caption_cfg.highlight_color, base_color=caption_cfg.base_color,
        outline_color=caption_cfg.outline_color, back_color=caption_cfg.back_color,
        position=caption_cfg.position, border_style=caption_cfg.border_style,
        outline_width=caption_cfg.outline_width, shadow_width=caption_cfg.shadow_width,
        emphasis_words=emphasis_words,
    )
    write_ass_file(ass_content, ass_path)
    burn_subtitles(source_video_path, ass_path, tmp_output_path)
    tmp_output_path.replace(final_output_path)
