"""End-to-end smoke test for the two-phase pipeline (analyze_video() then
render_selected()) WITHOUT needing a live Whisper model download (this
sandbox's network policy blocks huggingface.co; a real deployment on the
user's own machine won't have that restriction). A synthetic Transcript is
built by evenly distributing the known narration script across the source
video's real duration, then dropped into the job's transcript cache file so
analyze_video() picks it up instead of trying to transcribe for real — good
enough to prove the pipeline mechanics (candidate generation, review-style
selection/editing, style presets, rendering) are correct; real runs use
transcribe.py's actual word timestamps.
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.pipeline.render import analyze_video, render_selected
from app.pipeline.presets import get_aspect_preset, get_caption_preset
from app.pipeline.transcribe import Segment, Transcript, Word

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def build_fake_transcript(duration: float) -> Transcript:
    tokens = SCRIPT_TEXT.split()
    weights = [len(t) + (3 if t.endswith((".", "?", "!")) else 0.4) for t in tokens]
    total_weight = sum(weights)
    scale = duration / total_weight
    words = []
    t = 0.2
    for tok, w in zip(tokens, weights):
        dur = w * scale
        words.append(Word(word=tok, start=round(t, 3), end=round(t + dur * 0.75, 3), prob=0.95))
        t += dur
    seg = Segment(id=0, start=words[0].start, end=words[-1].end, text=SCRIPT_TEXT, words=words)
    return Transcript(language="en", duration=duration, segments=[seg])


if __name__ == "__main__":
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shorts_ai_test/source.mp4")
    job_dir = Path("/tmp/shorts_ai_test/job_offline2")
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    cfg.highlights.clip_count = 2
    cfg.highlights.min_duration_sec = 6
    cfg.highlights.max_duration_sec = 16
    cfg.highlights.use_llm = False

    duration = probe_duration(source)
    print(f"Source duration: {duration:.2f}s")
    transcript = build_fake_transcript(duration)

    # Seed the transcript cache so analyze_video()'s transcribe_video() call
    # loads this synthetic transcript instead of trying to run real Whisper.
    (job_dir / "transcript.json").write_text(json.dumps(transcript.to_json(), indent=2))

    # ---- Phase 1: analyze -------------------------------------------------
    print("\n=== PHASE 1: analyze_video() ===")
    analysis = analyze_video(source, job_dir, cfg, max_candidates=4)
    print(f"Found {len(analysis['candidates'])} candidates to review:")
    for c in analysis["candidates"]:
        print(f"  id={c['id']} [{c['start']:.1f}-{c['end']:.1f}] score={c['score']:.2f} title={c['title']!r}")
        print(f"    reasons: {c['reasons']}")

    assert analysis["candidates"], "expected at least one candidate"

    # ---- Simulate the review step: keep candidate 0 as-is, nudge candidate
    # 1's boundaries (as if the user dragged the trim handles), and give it a
    # custom title — proving render_selected() re-slices words by the EDITED
    # start/end rather than trusting the original candidate's word list. ----
    selections = [dict(analysis["candidates"][0])]
    if len(analysis["candidates"]) > 1:
        edited = dict(analysis["candidates"][1])
        edited["start"] = max(0.0, edited["start"] + 0.5)
        edited["end"] = edited["end"] - 0.5
        edited["title"] = "User-edited title"
        edited.pop("text", None)  # force render_selected to rebuild text from re-sliced words
        selections.append(edited)

    # Apply a non-default caption preset + aspect preset, exactly like
    # JobManager.start_render() does, to prove presets actually flow through
    # to the rendered output (not just sit unused in presets.py).
    style = get_caption_preset("boxed")
    cfg.captions.font = style["font"]
    cfg.captions.font_size = style["font_size"]
    cfg.captions.highlight_color = style["highlight_color"]
    cfg.captions.base_color = style["base_color"]
    cfg.captions.outline_color = style["outline_color"]
    cfg.captions.back_color = style["back_color"]
    cfg.captions.position = style["position"]
    cfg.captions.border_style = style["border_style"]
    cfg.captions.outline_width = style["outline_width"]
    cfg.captions.shadow_width = style["shadow_width"]

    aspect = get_aspect_preset("1:1")
    cfg.reframe.target_aspect = "1:1"
    cfg.reframe.target_width = aspect["width"]
    cfg.reframe.target_height = aspect["height"]

    # ---- Phase 2: render ----------------------------------------------------
    print("\n=== PHASE 2: render_selected() ===")
    print(f"Rendering {len(selections)} selected candidate(s) with caption_preset=boxed, aspect_preset=1:1")
    summary = render_selected(source, job_dir, cfg, selections)

    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2))

    out_dir = job_dir / "output"
    for r in summary["shorts"]:
        p = out_dir / r["file"]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "json", str(p)],
            capture_output=True, text=True, check=True,
        ).stdout
        print(f"{p.name}: {probe.strip()}")

    # Sanity checks on the things this rework specifically changed.
    for r in summary["shorts"]:
        p = out_dir / r["file"]
        import cv2
        cap = cv2.VideoCapture(str(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert (w, h) == (aspect["width"], aspect["height"]), f"expected {aspect['width']}x{aspect['height']}, got {w}x{h}"
    print("\nAspect preset applied correctly to rendered output dimensions. OK")
    assert summary["shorts"][-1]["title"] == "User-edited title", "edited title did not carry through"
    print("Edited title carried through render_selected(). OK")
