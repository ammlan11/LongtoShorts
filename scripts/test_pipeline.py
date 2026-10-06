"""Quick end-to-end smoke test against REAL transcription (needs network
access the first time, to download Whisper model weights — unlike
test_pipeline_offline.py, which fakes the transcript entirely). Not pytest,
just a runnable script that exercises both pipeline phases against the
synthetic test video and prints what happened, so failures are easy to see.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.pipeline.render import analyze_video, render_selected

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
# Speed up for a quick smoke test / short synthetic source.
cfg.transcription.model_size = "tiny"
cfg.highlights.clip_count = 2
cfg.highlights.min_duration_sec = 6
cfg.highlights.max_duration_sec = 20
cfg.highlights.use_llm = False  # no Ollama in this sandbox — test the pure-heuristic path

source = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shorts_ai_test/source.mp4")
job_dir = Path("/tmp/shorts_ai_test/job1")
job_dir.mkdir(parents=True, exist_ok=True)

print("=== PHASE 1: analyze_video() (real Whisper transcription) ===")
analysis = analyze_video(source, job_dir, cfg)
print(json.dumps(analysis, indent=2))

if not analysis["candidates"]:
    print("No candidates found — nothing to render.")
    sys.exit(0)

# Simulate the review step accepting every candidate as-is.
selections = analysis["candidates"]

print("\n=== PHASE 2: render_selected() ===")
summary = render_selected(source, job_dir, cfg, selections)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
