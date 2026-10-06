"""Regression test for a real bug: ffmpeg's filter-expression parser has a
hard nesting-depth limit (empirically ~100 chained if()s — see
_decimate_breakpoints()'s docstring in reframe.py). At the default 0.5s face-
tracking sample interval, any clip longer than ~50 seconds used to produce
100+ breakpoints and fail with exit code 234 (EINVAL) — silently breaking
reframing for most real-length shorts, since the highlight picker's own
target range goes up to 59s.

This test builds a 58-second synthetic source (deliberately past the old
50s cliff) and runs the real reframe pipeline against it end to end, so a
future change that removes or weakens the breakpoint decimation gets caught
immediately instead of only showing up on a user's own long clip.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.reframe import apply_reframe

DURATION = 58  # seconds — past the ~50s cliff that used to break this


def main():
    work_dir = Path("/tmp/shorts_ai_test/reframe_long_clip")
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "long_source.mp4"
    output = work_dir / "long_vertical.mp4"

    print(f"Generating a {DURATION}s synthetic source with a moving test pattern...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=size=1280x720:duration={DURATION}:rate=15",
            "-c:v", "libx264", "-preset", "ultrafast", str(source),
        ],
        capture_output=True, text=True, check=True,
    )

    print("Running apply_reframe() end to end (this is the real code path, not a mock)...")
    plan = apply_reframe(source, output, target_width=1080, target_height=1920, smoothing_window=15)

    assert output.exists() and output.stat().st_size > 0, "reframe produced no output file"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe == "1080,1920", f"expected 1080x1920, got {probe}"

    print(f"\nSUCCESS: {DURATION}s clip reframed without hitting ffmpeg's expression limit.")
    print(f"  axis={plan['axis']}  face detected={plan['found_face']}  output={probe}")


if __name__ == "__main__":
    main()
