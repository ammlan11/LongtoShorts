"""FastAPI app: upload a long video, review a shortlist of AI-picked
candidate moments, pick a caption/aspect style, then render just the ones
you want as Shorts-ready vertical clips — all processed locally.

The flow is two steps against the API (mirrored by the two-step frontend):
  1. POST /api/upload or /api/upload-url  -> job starts analyzing
     GET  /api/jobs/{id}                  -> poll until status=awaiting_review,
                                              then `candidates` is populated
  2. POST /api/jobs/{id}/render           -> submit the (edited) selection +
                                              style choice; job starts rendering
     GET  /api/jobs/{id}                  -> poll until status=done,
                                              then `summary.shorts` is populated

Run with:  python run.py
Then open: http://localhost:8000
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .jobs import JobManager
from .pipeline.postedit import PostEditError
from .pipeline.presets import ASPECT_PRESETS, CAPTION_PRESETS, DEFAULT_ASPECT_PRESET, DEFAULT_CAPTION_PRESET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

cfg = load_config()
jobs = JobManager(cfg)

app = FastAPI(title="Shorts AI")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/presets")
def presets():
    """Caption style + aspect ratio options for the review step's pickers."""
    return {
        "captions": [
            {"id": k, "label": v["label"], "description": v["description"]}
            for k, v in CAPTION_PRESETS.items()
        ],
        "default_caption_preset": DEFAULT_CAPTION_PRESET,
        "aspects": [{"id": k, "label": v["label"]} for k, v in ASPECT_PRESETS.items()],
        "default_aspect_preset": DEFAULT_ASPECT_PRESET,
    }


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    clip_count: int | None = Form(None),
    captions_enabled: bool = Form(True),
    use_llm: bool = Form(True),
):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest_name = f"{uuid.uuid4().hex[:12]}{suffix}"
    dest_path = cfg.uploads_dir / dest_name
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Note: no "overlays_enabled" override here -- the stat-card graphic
    # overlays are retired (low-contrast, hard to read; see OverlaysConfig
    # in app/config.py), so it's always off regardless of input.
    job = jobs.create_job(
        dest_path,
        overrides={
            "clip_count": clip_count,
            "captions_enabled": captions_enabled,
            "use_llm": use_llm,
        },
    )
    return {"job_id": job.id}


@app.post("/api/upload-url")
async def upload_url(
    url: str = Form(...),
    clip_count: int | None = Form(None),
    captions_enabled: bool = Form(True),
    use_llm: bool = Form(True),
):
    url = url.strip()
    if not url:
        raise HTTPException(400, "Please paste a video URL")

    job = jobs.create_job_from_url(
        url,
        overrides={
            "clip_count": clip_count,
            "captions_enabled": captions_enabled,
            "use_llm": use_llm,
        },
    )
    return {"job_id": job.id}


class Selection(BaseModel):
    id: Optional[int] = None
    start: float
    end: float
    title: Optional[str] = None
    score: Optional[float] = None
    reasons: Optional[list[str]] = None
    text: Optional[str] = None
    most_replayed: Optional[bool] = None
    is_recap: Optional[bool] = None


class CaptionOverrides(BaseModel):
    position: Optional[str] = None
    font_size: Optional[int] = None
    text_color: Optional[str] = None
    highlight_color: Optional[str] = None
    background_box: Optional[bool] = None


class RenderRequest(BaseModel):
    selections: list[Selection]
    caption_preset: Optional[str] = None
    aspect_preset: Optional[str] = None
    caption_overrides: Optional[CaptionOverrides] = None


@app.post("/api/jobs/{job_id}/render")
def render(job_id: str, body: RenderRequest):
    """Step 2: the user has reviewed (and maybe edited) the candidates from
    analyze_video() and is ready to actually render some of them."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "awaiting_review":
        raise HTTPException(409, f"job is '{job.status}', not ready to render yet")
    if not body.selections:
        raise HTTPException(400, "pick at least one candidate to render")

    selections = [s.model_dump(exclude_none=True) for s in body.selections]
    overrides = body.caption_overrides.model_dump(exclude_none=True) if body.caption_overrides else None
    started = jobs.start_render(
        job_id, selections,
        caption_preset=body.caption_preset,
        aspect_preset=body.aspect_preset,
        caption_overrides=overrides,
    )
    if started is None:
        raise HTTPException(409, "could not start render")
    return {"job_id": job.id, "status": started.status}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JSONResponse(
        {
            "id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "error": job.error,
            "candidates": (job.analysis or {}).get("candidates") if job.analysis else None,
            "source_duration": (job.analysis or {}).get("source_duration") if job.analysis else None,
            "summary": job.summary,
            "timing": job.timing,
        }
    )


class TrimRequest(BaseModel):
    trim_start: float = 0.0
    trim_end: float = 0.0


class UpdateShortRequest(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None


class EmphasisRequest(BaseModel):
    words: list[str] = []


@app.post("/api/jobs/{job_id}/shorts/{index}/trim")
def trim_short(job_id: str, index: int, body: TrimRequest):
    """Step 3 (results page): quick-trim an already-rendered short in place.
    Fast -- just cuts the finished file -- but doesn't re-time any captions
    or overlays to the new boundaries (see pipeline/postedit.py)."""
    if body.trim_start < 0 or body.trim_end < 0:
        raise HTTPException(400, "trim amounts must be >= 0")
    try:
        meta = jobs.trim_short(job_id, index, body.trim_start, body.trim_end)
    except PostEditError as e:
        raise HTTPException(400, str(e))
    return meta


@app.post("/api/jobs/{job_id}/shorts/{index}/update")
def update_short(job_id: str, index: int, body: UpdateShortRequest):
    """Step 3: save an edited title and/or upload-ready hashtag/keyword tags
    for a rendered short (also used to apply a picked AI caption suggestion)."""
    try:
        meta = jobs.update_short(job_id, index, title=body.title, tags=body.tags)
    except PostEditError as e:
        raise HTTPException(400, str(e))
    return meta


@app.post("/api/jobs/{job_id}/shorts/{index}/emphasis")
def emphasize_short(job_id: str, index: int, body: EmphasisRequest):
    """Step 3: re-burn this short's captions with the given words always
    highlighted, wherever they occur in the clip."""
    try:
        meta = jobs.apply_short_emphasis(job_id, index, body.words)
    except PostEditError as e:
        raise HTTPException(400, str(e))
    return meta


@app.post("/api/jobs/{job_id}/shorts/{index}/suggest-captions")
def suggest_short_captions(job_id: str, index: int):
    """Step 3: ask the local LLM (or, if unavailable, a built-in heuristic)
    for a few punchy caption/hashtag options based on this short's own
    transcript. Doesn't change anything -- apply a pick via the /update
    endpoint above."""
    try:
        return {"suggestions": jobs.suggest_short_captions(job_id, index)}
    except PostEditError as e:
        raise HTTPException(400, str(e))


@app.get("/api/jobs/{job_id}/download/{filename}")
def download(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    path = job.job_dir / "output" / filename
    if not path.exists() or path.parent != job.job_dir / "output":
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")
