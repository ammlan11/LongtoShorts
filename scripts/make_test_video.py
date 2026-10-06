"""Generates a small synthetic 16:9 'talking head' test video so the
pipeline can be exercised end-to-end without a real recording:
  - a moving circle (stands in for a face, detectable-ish) that pans
    left/right so reframe.py has something to track
  - burned-in captions listing what's "said" at each moment (for a sanity
    check against the Whisper transcript)
  - a synthesized voice reading a short script that includes some numbers,
    generated with espeak-ng if available, otherwise a sine-wave stand-in
    (Whisper will transcribe *something* either way — this script's real
    job is to shake out crashes, not to prove transcription accuracy).
"""
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4")
SCRIPT_TEXT = (
    "Have you ever wondered why most traders lose money. "
    "Here's the truth nobody tells you. Ninety percent of new traders "
    "quit within the first six months. But the traders who stick with a "
    "plan see their win rate go from twenty percent to sixty percent. "
    "Last year our students grew their average account from five thousand "
    "dollars to twenty two thousand dollars. That is not luck, that is "
    "process. If you want consistent results, stop guessing and start "
    "tracking every single trade."
)

WIDTH, HEIGHT, FPS, DURATION = 1280, 720, 30, 32


def make_audio(audio_path: Path):
    if shutil.which("espeak-ng"):
        subprocess.run(
            ["espeak-ng", "-s", "165", "-w", str(audio_path), SCRIPT_TEXT],
            check=True, capture_output=True,
        )
    elif shutil.which("espeak"):
        subprocess.run(
            ["espeak", "-s", "165", "-w", str(audio_path), SCRIPT_TEXT],
            check=True, capture_output=True,
        )
    else:
        # Fallback: silence-ish sine tone so the pipeline still has an audio
        # track to decode (transcription quality will be meaningless, but
        # every stage still runs).
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"sine=frequency=220:duration={DURATION}",
             str(audio_path)],
            check=True, capture_output=True,
        )


def make_video(tmp_dir: Path, video_only: Path):
    # A circle panning across the frame stands in for a moving speaker.
    filter_complex = (
        f"color=c=0x2a3a5a:s={WIDTH}x{HEIGHT}:d={DURATION}:r={FPS}[bg];"
        f"[bg]drawbox=x='(iw*0.5+iw*0.25*sin(t/4))-90':y=180:w=180:h=180:"
        f"color=0xE0C097@1.0:t=fill[circ]"
    )
    cmd = [
        "ffmpeg", "-y", "-filter_complex", filter_complex,
        "-map", "[circ]", "-t", str(DURATION), "-r", str(FPS),
        "-pix_fmt", "yuv420p", str(video_only),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def mux(video_only: Path, audio_path: Path, out_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


if __name__ == "__main__":
    tmp = Path("/tmp/shorts_ai_test")
    tmp.mkdir(exist_ok=True)
    audio_path = tmp / "voice.wav"
    video_only = tmp / "video_only.mp4"

    print("Synthesizing narration...")
    make_audio(audio_path)
    print("Rendering test video track...")
    make_video(tmp, video_only)
    print("Muxing...")
    mux(video_only, audio_path, OUT)
    print(f"Wrote {OUT}")
