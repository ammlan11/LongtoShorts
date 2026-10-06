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
