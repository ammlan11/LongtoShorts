"""End-to-end HTTP test for the results-page post-render editing endpoints
(trim / update / emphasis / suggest-captions) added to app/main.py, driven
through FastAPI's TestClient against the real app + JobManager + routes --
complements test_postedit.py (which calls JobManager methods directly) and
test_api_e2e.py (which covers the upload -> review -> render flow but not
these newer endpoints).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_api_e2e as e2e  # reuse its fake-transcript + transcribe patch


def job_dir_for(client, job_id):
    from app.main import jobs
    return jobs.get(job_id).job_dir


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shorts_ai_test/source.mp4")
    duration = e2e.probe_duration(source)
    fake_transcript = e2e.build_fake_transcript(duration)

    import app.pipeline.render as render_mod

    def fake_transcribe_video(video_path, cache_path=None, progress_cb=None, **kwargs):
        if cache_path:
            import json
            cache_path = Path(cache_path)
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

    with open(source, "rb") as f:
        r = client.post(
            "/api/upload",
            files={"file": ("source.mp4", f, "video/mp4")},
            data={"clip_count": "3", "captions_enabled": "true", "use_llm": "false"},
        )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    for _ in range(60):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("awaiting_review", "error"):
            break
        time.sleep(0.3)
    assert job["status"] == "awaiting_review", job

    selections = job["candidates"][:1]
    r = client.post(f"/api/jobs/{job_id}/render", json={"selections": selections})
    assert r.status_code == 200, r.text

    for _ in range(120):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert job["status"] == "done", job
    short = job["summary"]["shorts"][0]
    index = short["index"]
    print(f"Rendered short index={index} file={short['file']} duration={short['duration']}")

    print("\n=== POST /suggest-captions ===")
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/suggest-captions")
    assert r.status_code == 200, r.text
    suggestions = r.json()["suggestions"]
    assert suggestions and all(s.get("caption") and s.get("hashtags") for s in suggestions)
    print(suggestions)

    print("\n=== POST /update (title + tags) ===")
    r = client.post(
        f"/api/jobs/{job_id}/shorts/{index}/update",
        json={"title": suggestions[0]["caption"], "tags": [h.lstrip("#") for h in suggestions[0]["hashtags"]]},
    )
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["title"] == suggestions[0]["caption"]
    assert meta["tags"] == [h.lstrip("#") for h in suggestions[0]["hashtags"]]
    print(meta)

    print("\n=== POST /emphasis ===")
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/emphasis", json={"words": ["profit"]})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["emphasis_words"] == ["profit"]
    print(meta)

    print("\n=== POST /trim ===")
    orig_duration = meta["duration"]
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/trim", json={"trim_start": 1.0, "trim_end": 0.0})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert abs(meta["duration"] - (orig_duration - 1.0)) < 0.2, meta
    print(meta)

    print("\n=== POST /captions (remove, then restore), after a trim ===")
    def dur(path):
        return e2e.probe_duration(path)
    out = Path(job_dir_for(client, job_id)) / "output" / meta["file"]
    def grab(tag):
        import subprocess, numpy as np
        png = f"/tmp/frame_{tag}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "3", "-i", str(out), "-frames:v", "1", png], check=True)
        from PIL import Image
        a = np.asarray(Image.open(png).convert("L"), dtype=float)
        h = a.shape[0]
        return a[int(h * 0.60):int(h * 0.85)]  # lower-third caption band
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/captions", json={"enabled": False})
    assert r.status_code == 200, r.text
    off = r.json()
    band_off = grab("off")
    assert off["captions_enabled"] is False
    assert abs(dur(out) - off["duration"]) < 0.5, (dur(out), off["duration"])  # trim kept
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/captions", json={"enabled": False})
    assert r.status_code == 200 and r.json()["captions_enabled"] is False  # idempotent
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/captions", json={"enabled": True})
    assert r.status_code == 200, r.text
    on = r.json()
    assert on["captions_enabled"] is True and on["emphasis_words"] == ["profit"]
    assert abs(dur(out) - on["duration"]) < 0.5, (dur(out), on["duration"])
    band_on = grab("on")
    import numpy as np
    assert np.abs(band_on - band_off).mean() > 1.0, "captions band looks identical on vs off"
    print("captions off/on OK; durations stay", on["duration"], "band diff %.2f" % np.abs(band_on - band_off).mean())

    print("\n=== POST /trim with an invalid (negative) amount -> expect 400 ===")
    r = client.post(f"/api/jobs/{job_id}/shorts/{index}/trim", json={"trim_start": -5.0, "trim_end": 0.0})
    assert r.status_code == 400, r.text
    print("Correctly rejected:", r.json())

    print("\n=== GET job download after trim still serves a valid file ===")
    r = client.get(f"/api/jobs/{job_id}/download/{meta['file']}")
    assert r.status_code == 200
    assert len(r.content) > 1000

    print("\nALL POST-EDIT HTTP API CHECKS PASSED")


if __name__ == "__main__":
    main()
