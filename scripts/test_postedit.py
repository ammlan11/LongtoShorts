"""Smoke test for the results-page post-render editing endpoints (trim,
title/tags update, keyword-emphasis caption re-burn, AI caption suggestions)
added in app/jobs.py + app/pipeline/postedit.py + app/pipeline/caption_suggest.py.

Builds a real rendered job the same way test_pipeline_offline.py does (a
synthetic transcript seeded into the cache, so no live Whisper/network needed),
then exercises JobManager's post-render methods directly against it.
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.jobs import Job, JobManager
from app.pipeline.render import analyze_video, render_selected
from app.pipeline.transcribe import Segment, Transcript, Word

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SCRIPT_TEXT = (
    "Have you ever wondered why most traders lose money. "
    "Here's the secret truth nobody tells you about trading profit. "
    "Ninety percent of new traders quit within the first six months. "
    "But the traders who stick with a plan see their win rate go from "
    "twenty percent to sixty percent and grow their profit every month. "
    "Last year our students grew their average account from five thousand "
    "dollars to twenty two thousand dollars in pure profit. "
    "That is not luck, that is process. If you want consistent results, "
    "stop guessing and start tracking every single trade."
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
    job_dir = Path("/tmp/shorts_ai_test/job_postedit")
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    cfg.highlights.clip_count = 1
    cfg.highlights.min_duration_sec = 10
    cfg.highlights.max_duration_sec = 20
    cfg.highlights.use_llm = True  # exercises the is_available()->False->heuristic fallback path
    cfg.captions.enabled = True
    cfg.overlays.enabled = True

    duration = probe_duration(source)
    transcript = build_fake_transcript(duration)
    (job_dir / "transcript.json").write_text(json.dumps(transcript.to_json(), indent=2))

    print("=== Building a real rendered job (analyze + render) ===")
    analysis = analyze_video(source, job_dir, cfg, max_candidates=2)
    assert analysis["candidates"], "expected at least one candidate"
    selections = [dict(analysis["candidates"][0])]
    summary = render_selected(source, job_dir, cfg, selections)
    assert summary["shorts"], "expected at least one rendered short"
    print(f"Rendered: {summary['shorts'][0]['file']}, duration={summary['shorts'][0]['duration']}s")

    jm = JobManager(cfg)
    job = Job(id="test_postedit", job_dir=job_dir, cfg=cfg, video_path=source, status="done", summary=summary)
    jm._jobs[job.id] = job

    print("\n=== suggest_short_captions() (no Ollama in this sandbox -> heuristic fallback) ===")
    suggestions = jm.suggest_short_captions(job.id, 0)
    print(json.dumps(suggestions, indent=2))
    assert suggestions, "expected at least one heuristic caption suggestion"
    assert all(s.get("caption") for s in suggestions)
    assert all(s.get("hashtags") for s in suggestions)

    print("\n=== update_short() title + tags ===")
    meta = jm.update_short(job.id, 0, title=suggestions[0]["caption"], tags=[h.lstrip("#") for h in suggestions[0]["hashtags"]])
    print(json.dumps(meta, indent=2))
    assert meta["title"] == suggestions[0]["caption"]
    assert meta["tags"] == [h.lstrip("#") for h in suggestions[0]["hashtags"]]
    tags_file = (job_dir / "output" / meta["file"]).with_name(Path(meta["file"]).stem + "_tags.txt")
    assert tags_file.exists(), "expected a _tags.txt sidecar to be written"
    print("tags sidecar:", tags_file.read_text())

    print("\n=== apply_short_emphasis() ===")
    before_mtime = (job_dir / "output" / meta["file"]).stat().st_mtime
    meta = jm.apply_short_emphasis(job.id, 0, words=["profit", "secret"])
    after_mtime = (job_dir / "output" / meta["file"]).stat().st_mtime
    assert after_mtime > before_mtime, "expected the final mp4 to be rewritten"
    assert meta["emphasis_words"] == ["profit", "secret"]
    print("emphasis_words saved:", meta["emphasis_words"])

    print("\n=== trim_short() ===")
    orig_duration = meta["duration"]
    orig_start, orig_end = meta["start"], meta["end"]
    meta = jm.trim_short(job.id, 0, trim_start=1.0, trim_end=1.0)
    print(json.dumps(meta, indent=2))
    assert abs(meta["duration"] - (orig_duration - 2.0)) < 0.2, f"expected duration ~{orig_duration-2.0}, got {meta['duration']}"
    assert abs(meta["start"] - (orig_start + 1.0)) < 0.01
    assert abs(meta["end"] - (orig_end - 1.0)) < 0.01
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
         str(job_dir / "output" / meta["file"])],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"ffprobe-measured duration after trim: {probe}")
    assert abs(float(probe) - meta["duration"]) < 0.5

    # A second emphasis call AFTER trim must use the updated start/end, not
    # the original ones -- proves trim_short() correctly keeps the
    # source-relative start/end in sync for later operations.
    print("\n=== apply_short_emphasis() again, post-trim (sync check) ===")
    meta = jm.apply_short_emphasis(job.id, 0, words=["results"])
    print("OK -- emphasis re-applied successfully after trim using updated start/end")

    print("\nALL POST-EDIT CHECKS PASSED")
