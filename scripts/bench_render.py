"""Times the render stage (cut + 9:16 reframe + captions) for ONE real clip
under different settings, so speed-ups can be measured instead of guessed.

    python scripts/bench_render.py storage/jobs/<job_id> [--candidate 0]

Uses the job's own analysis.json (source name, candidate window) and
transcript.json (caption words). Output goes to a temp folder; nothing in the
job is touched. Prints one line per mode:  mode  seconds  output_duration.
"""
import argparse, json, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load_config
from app.pipeline import ffmpeg_util as fu
from app.pipeline.highlights import Candidate
from app.pipeline.render import render_clip
from app.pipeline.reframe import probe_video
from app.pipeline.transcribe import Transcript
import dataclasses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--candidate", type=int, default=0)
    ap.add_argument("--modes", default="old,new")
    a = ap.parse_args()

    cfg = load_config()
    job = Path(a.job_dir)
    analysis = json.loads((job / "analysis.json").read_text())
    c = analysis["candidates"][a.candidate]
    tr = Transcript.from_json(json.loads((job / "transcript.json").read_text()))
    words = [w for w in tr.flat_words() if w.start < c["end"] and w.end > c["start"]]
    fields = {f.name for f in dataclasses.fields(Candidate)}
    cand = Candidate(**{**{k: v for k, v in c.items() if k in fields}, "words": words})
    source = cfg.uploads_dir / analysis["source"]
    print(f"source={source.name} window={c['start']:.1f}-{c['end']:.1f}s ({c['duration']:.0f}s) words={len(words)}")

    modes = {
        # the pipeline as it was: 3 separate libx264 encodes + full-res face detection
        "old": dict(single_pass=False, encoder="libx264"),
        # new default: one pass, one encode (libx264 on every OS)
        "new": dict(single_pass=True, encoder="auto"),
        # same, but with the Mac hardware encoder (opt-in; measured slower)
        "new-vt": dict(single_pass=True, encoder="videotoolbox"),
    }
    for name in a.modes.split(","):
        m = modes[name]
        cfg.render.single_pass = m["single_pass"]
        fu.set_encoder_preference(m["encoder"])
        with tempfile.TemporaryDirectory(prefix="bench_render_") as tmp:
            t = time.perf_counter()
            meta = render_clip(source, cand, Path(tmp) / "out", Path(tmp) / "work", 0, cfg)
            secs = time.perf_counter() - t
            dur = probe_video(Path(tmp) / "out" / meta["file"]).duration
        print(f"RESULT mode={name:9s} encoder={fu.resolve_encoder():12s} seconds={secs:6.1f} out_duration={dur:.1f}s", flush=True)


if __name__ == "__main__":
    main()
