"""End-to-end test of the actual HTTP API (main.py + jobs.py) for the new
two-phase analyze -> review -> render flow, using FastAPI's TestClient so it
drives the real ASGI app, the real JobManager background threads, and the
real route handlers — not just the pipeline functions directly (that's what
test_pipeline_offline.py already covers).

The only thing faked is transcribe_video() itself (patched onto the render
module), since this sandbox's network policy blocks the Whisper model
download from huggingface.co; a real deployment won't need this patch.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def build_fake_transcript(duration: float):
    from app.pipeline.transcribe import Segment, Transcript, Word

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


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shorts_ai_test/source.mp4")
    duration = probe_duration(source)
    fake_transcript = build_fake_transcript(duration)

    # Patch transcribe_video as used inside render.py (it imported the name
    # directly, so patch it there, not on the transcribe module).
    import app.pipeline.render as render_mod

    def fake_transcribe_video(video_path, cache_path=None, progress_cb=None, **kwargs):
        if cache_path:
            from pathlib import Path as P
            import json
            cache_path = P(cache_path)
            if cache_path.exists():
                from app.pipeline.transcribe import Transcript
                return Transcript.from_json(json.loads(cache_path.read_text()))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(fake_transcript.to_json(), indent=2))
        return fake_transcript

    render_mod.transcribe_video = fake_transcribe_video

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # ---- /api/presets -------------------------------------------------------
    r = client.get("/api/presets")
    assert r.status_code == 200, r.text
    presets = r.json()
    print("Presets:", [p["id"] for p in presets["captions"]], [p["id"] for p in presets["aspects"]])
    assert "clean-white" in [p["id"] for p in presets["captions"]]
    assert "4:5" in [p["id"] for p in presets["aspects"]]

    # ---- step 1: upload -------------------------------------------------
    with open(source, "rb") as f:
        r = client.post(
            "/api/upload",
            files={"file": ("source.mp4", f, "video/mp4")},
            data={"clip_count": "4", "captions_enabled": "true", "use_llm": "false"},
        )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    print(f"\nJob created: {job_id}")

    # ---- poll until awaiting_review ------------------------------------
    for _ in range(60):
        r = client.get(f"/api/jobs/{job_id}")
        job = r.json()
        print(f"  status={job['status']} stage={job['stage']} progress={job['progress']:.2f}")
        if job["status"] in ("awaiting_review", "error"):
            break
        time.sleep(0.3)
    assert job["status"] == "awaiting_review", f"job ended in unexpected status: {job}"
    candidates = job["candidates"]
    assert candidates, "expected candidates from analyze phase"
    print(f"\nGot {len(candidates)} candidates for review:")
    for c in candidates:
        print(f"  id={c['id']} [{c['start']}-{c['end']}] {c['title']!r} score={c['score']}")

    # ---- step 2: simulate the review step (edit one candidate) ---------
    selections = candidates[:2]
    selections[0] = dict(selections[0])
    selections[0]["title"] = "Edited via API test"
    selections[0]["start"] = max(0.0, selections[0]["start"] + 0.3)

    r = client.post(
        f"/api/jobs/{job_id}/render",
        json={"selections": selections, "caption_preset": "clean-white", "aspect_preset": "4:5"},
    )
    assert r.status_code == 200, r.text
    print("\nRender started:", r.json())

    # ---- poll until done -------------------------------------------------
    for _ in range(120):
        r = client.get(f"/api/jobs/{job_id}")
        job = r.json()
        print(f"  status={job['status']} stage={job['stage']} progress={job['progress']:.2f}")
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert job["status"] == "done", f"render ended in unexpected status: {job}"

    summary = job["summary"]
    assert len(summary["shorts"]) == len(selections), "rendered count != selected count"
    assert summary["shorts"][0]["title"] == "Edited via API test", "edited title did not survive the API round-trip"
    print("\nRendered shorts:")
    for s in summary["shorts"]:
        print(f"  {s['file']}  title={s['title']!r}  duration={s['duration']}s  overlays={s['overlay_count']}")

    # ---- download + verify aspect preset applied -------------------------
    r = client.get(f"/api/jobs/{job_id}/download/{summary['shorts'][0]['file']}")
    assert r.status_code == 200
    out_path = Path("/tmp/shorts_ai_test/api_e2e_download.mp4")
    out_path.write_bytes(r.content)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(out_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\nDownloaded clip dimensions: {probe}")
    assert probe == "1080,1350", f"expected 4:5 (1080x1350), got {probe}"

    print("\nALL API E2E CHECKS PASSED")


if __name__ == "__main__":
    main()
