"""Minimal in-memory job manager. This is a single-user local tool — no
database, no auth, no multi-tenancy. Each job now runs in two background-
thread phases instead of one:

  1. analyze  — download (if from a URL) + transcribe + find candidate
                highlights, then stop and wait ("awaiting_review").
  2. render   — triggered separately once the user has reviewed/edited the
                candidate list and picked a caption/aspect style; only then
                do the (slow) ffmpeg passes actually run.

Splitting it this way is what lets the UI show "here are 5 possible shorts,
pick/edit the ones you want" before spending render time on any of them.
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import AppConfig
from .pipeline import postedit
from .pipeline.caption_suggest import suggest_captions
from .pipeline.postedit import PostEditError
from .pipeline.presets import CUSTOM_PRESET_ID, get_aspect_preset, get_caption_preset, hex_to_ass
from .pipeline.render import analyze_video, render_selected
from .pipeline.transcribe import Transcript

logger = logging.getLogger("shorts_ai.jobs")

# queued -> downloading -> analyzing -> awaiting_review -> rendering -> done
#                                                                     \-> error
STATUSES = (
    "queued", "downloading", "analyzing", "awaiting_review",
    "rendering", "done", "error",
)


@dataclass
class Job:
    id: str
    job_dir: Path
    cfg: AppConfig
    video_path: Optional[Path] = None
    status: str = "queued"
    stage: str = ""
    progress: float = 0.0
    error: Optional[str] = None
    analysis: Optional[dict[str, Any]] = None   # candidates, from analyze_video()
    summary: Optional[dict[str, Any]] = None    # final shorts, from render_selected()
    heatmap: Optional[list[dict[str, Any]]] = None  # YouTube "most replayed" data, if any
    # Per-stage wall-clock seconds (download/transcribing/finding_highlights),
    # filled in as the job runs — surfaced in the API response so a slow run
    # can actually be diagnosed (which stage ate the time) instead of guessed
    # at from a single "it took 30 minutes" report.
    timing: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class JobManager:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ---- phase 1: analyze -------------------------------------------------

    def create_job(self, video_path: Path, overrides: dict[str, Any] | None = None) -> Job:
        job = self._new_job()
        job.video_path = video_path
        cfg = self._apply_overrides(overrides)
        job.cfg = cfg
        thread = threading.Thread(target=self._run_analyze, args=(job,), daemon=True)
        thread.start()
        return job

    def create_job_from_url(self, url: str, overrides: dict[str, Any] | None = None) -> Job:
        """Same as create_job, but the source video is downloaded first
        (yt-dlp) — used by the 'paste a link' path in the UI."""
        job = self._new_job()
        job.stage = "downloading"
        cfg = self._apply_overrides(overrides)
        job.cfg = cfg
        thread = threading.Thread(target=self._run_download_then_analyze, args=(job, url), daemon=True)
        thread.start()
        return job

    def _new_job(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.cfg.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, job_dir=job_dir, cfg=self.cfg)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def _apply_overrides(self, overrides: dict[str, Any] | None) -> AppConfig:
        cfg = self.cfg
        if not overrides:
            return cfg
        cfg = copy.deepcopy(cfg)
        if overrides.get("clip_count"):
            cfg.highlights.clip_count = int(overrides["clip_count"])
        if "captions_enabled" in overrides:
            cfg.captions.enabled = bool(overrides["captions_enabled"])
        # No "overlays_enabled" override: stat-card overlays are retired
        # (see OverlaysConfig in app/config.py) and always stay off.
        if "use_llm" in overrides:
            cfg.highlights.use_llm = bool(overrides["use_llm"])
        return cfg

    def _run_download_then_analyze(self, job: Job, url: str) -> None:
        job.status = "downloading"
        job.stage = "downloading"
        job.progress = 0.0
        try:
            from .pipeline.youtube import download_video

            def dl_progress(pct: float):
                job.progress = pct

            result = download_video(url, self.cfg.uploads_dir, progress_cb=dl_progress)
            job.video_path = result.path
            job.heatmap = result.heatmap
            job.timing["download_seconds"] = round(result.download_seconds, 1)
            logger.info("Job %s: download took %.1fs", job.id, result.download_seconds)
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            logger.error("Job %s failed to download %s: %s", job.id, url, e)
            job.status = "error"
            job.error = str(e)
            return
        self._run_analyze(job)

    def _run_analyze(self, job: Job) -> None:
        job.status = "analyzing"
        # Tracks wall-clock time per named stage (transcribing,
        # finding_highlights, ...) by watching for the stage name to change
        # between progress_cb calls — gives concrete per-stage numbers in
        # job.timing without needing analyze_video() itself to know about
        # timing at all.
        stage_tracker = {"stage": None, "started": None}

        def _flush_stage(now: float) -> None:
            if stage_tracker["stage"] is not None:
                key = f"{stage_tracker['stage']}_seconds"
                job.timing[key] = round(job.timing.get(key, 0.0) + (now - stage_tracker["started"]), 1)

        try:
            def progress_cb(stage: str, pct: float):
                now = time.monotonic()
                if stage_tracker["stage"] != stage:
                    _flush_stage(now)
                    stage_tracker["stage"] = stage
                    stage_tracker["started"] = now
                job.stage = stage
                job.progress = pct

            analysis = analyze_video(
                job.video_path, job.job_dir, job.cfg, progress_cb=progress_cb, heatmap=job.heatmap
            )
            _flush_stage(time.monotonic())
            analysis["timing"] = dict(job.timing)
            total = sum(job.timing.values())
            logger.info("Job %s: stage timing %s (total %.1fs)", job.id, job.timing, total)
            job.analysis = analysis
            job.status = "awaiting_review"
            job.stage = "awaiting_review"
            job.progress = 1.0
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            logger.error("Job %s failed to analyze: %s\n%s", job.id, e, traceback.format_exc())
            job.status = "error"
            job.error = str(e)

    # ---- phase 2: render ----------------------------------------------------

    def start_render(
        self,
        job_id: str,
        selections: list[dict[str, Any]],
        caption_preset: str | None = None,
        aspect_preset: str | None = None,
        caption_overrides: dict[str, Any] | None = None,
    ) -> Optional[Job]:
        """Kicks off the render phase for a job that's sitting in
        'awaiting_review'. `selections` is the (possibly user-edited) list of
        candidates to actually render — see render_selected() for the shape.
        Returns None if the job doesn't exist or isn't ready to be rendered.

        `caption_overrides` is the "Custom" style panel's field-by-field
        tweaks (position/font_size/text_color/highlight_color/background_box)
        — applied on top of whichever named preset was picked (or the
        default one, if caption_preset is the "custom" sentinel / omitted),
        so picking "Custom" and only moving the font-size slider doesn't
        reset everything else to defaults."""
        job = self._jobs.get(job_id)
        if job is None or job.status != "awaiting_review":
            return None
        if not selections:
            return None

        cfg = copy.deepcopy(job.cfg)
        if caption_preset and caption_preset != CUSTOM_PRESET_ID:
            style = get_caption_preset(caption_preset)
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
            cfg.captions.preset = caption_preset

        if caption_overrides:
            co = caption_overrides
            if co.get("position") in ("lower_third", "center", "top"):
                cfg.captions.position = co["position"]
            if co.get("font_size"):
                try:
                    cfg.captions.font_size = max(24, min(140, int(co["font_size"])))
                except (TypeError, ValueError):
                    pass
            if co.get("text_color"):
                cfg.captions.base_color = hex_to_ass(co["text_color"])
            if co.get("highlight_color"):
                cfg.captions.highlight_color = hex_to_ass(co["highlight_color"])
            if "background_box" in co:
                if co["background_box"]:
                    cfg.captions.border_style = 3
                    cfg.captions.outline_width = 0
                    cfg.captions.shadow_width = 0
                    cfg.captions.back_color = "&HD0161616"
                else:
                    cfg.captions.border_style = 1
                    cfg.captions.outline_width = 5
                    cfg.captions.shadow_width = 2
            cfg.captions.preset = CUSTOM_PRESET_ID

        if aspect_preset:
            aspect = get_aspect_preset(aspect_preset)
            cfg.reframe.target_aspect = aspect_preset
            cfg.reframe.target_width = aspect["width"]
            cfg.reframe.target_height = aspect["height"]
        job.cfg = cfg

        thread = threading.Thread(target=self._run_render, args=(job, selections), daemon=True)
        thread.start()
        return job

    def _run_render(self, job: Job, selections: list[dict[str, Any]]) -> None:
        job.status = "rendering"
        job.progress = 0.0
        try:
            def progress_cb(stage: str, pct: float):
                job.stage = stage
                job.progress = pct

            summary = render_selected(job.video_path, job.job_dir, job.cfg, selections, progress_cb=progress_cb)
            job.summary = summary
            job.status = "done"
            job.progress = 1.0
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            logger.error("Job %s failed to render: %s\n%s", job.id, e, traceback.format_exc())
            job.status = "error"
            job.error = str(e)

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    # ---- phase 3: post-render editing (results page) -----------------------
    #
    # These act on a job that's already "done" -- they touch the finished
    # short(s) directly rather than re-running the full pipeline, which is
    # what makes them fast. See pipeline/postedit.py for the tradeoffs that
    # come with that (trimming doesn't re-time captions/overlays).

    def _get_short(self, job_id: str, index: int) -> tuple[Optional[Job], Optional[dict]]:
        job = self._jobs.get(job_id)
        if job is None or job.status != "done" or not job.summary:
            return None, None
        meta = next((s for s in job.summary["shorts"] if s["index"] == index), None)
        return job, meta

    def _persist_short_metadata(self, job: Job, meta: dict) -> None:
        """Writes the updated per-short metadata dict back to its own
        short_N.json next to the video, and leaves job.summary already
        updated in place (the caller mutates the same dict object that's
        inside job.summary["shorts"], so no separate write-back is needed
        there)."""
        json_path = (job.job_dir / "output" / meta["file"]).with_suffix(".json")
        json_path.write_text(json.dumps(meta, indent=2))

    def trim_short(self, job_id: str, index: int, trim_start: float, trim_end: float) -> dict:
        job, meta = self._get_short(job_id, index)
        if job is None or meta is None:
            raise PostEditError("Short not found (job isn't done, or no such clip index).")

        final_path = job.job_dir / "output" / meta["file"]
        if not final_path.exists():
            raise PostEditError("That short's video file is missing on disk.")
        tmp_path = final_path.with_suffix(".trimtmp.mp4")

        new_duration = postedit.trim_rendered_clip(
            final_path, tmp_path, final_path,
            trim_start=trim_start, trim_end=trim_end, current_duration=meta["duration"],
        )
        # Keep the metadata's source-video start/end in sync with the trim so
        # a later "emphasize keywords" re-caption (which looks up transcript
        # words by this start/end) correctly excludes the trimmed-off parts.
        meta["start"] = round(meta["start"] + max(0.0, trim_start), 2)
        meta["end"] = round(meta["end"] - max(0.0, trim_end), 2)
        meta["duration"] = round(new_duration, 2)
        self._persist_short_metadata(job, meta)
        logger.info("Job %s short %d trimmed to %.1fs", job_id, index, new_duration)
        return meta

    def update_short(
        self, job_id: str, index: int, title: str | None = None, tags: list[str] | None = None
    ) -> dict:
        """Updates the title and/or hashtag/keyword tags saved alongside a
        rendered short. Tags are also written to a small sidecar .txt file
        next to the video so they're easy to copy-paste when uploading."""
        job, meta = self._get_short(job_id, index)
        if job is None or meta is None:
            raise PostEditError("Short not found (job isn't done, or no such clip index).")

        if title is not None:
            meta["title"] = title.strip()[:120]
        if tags is not None:
            cleaned = []
            for t in tags:
                t = t.strip().lstrip("#").replace(" ", "")
                if t and t not in cleaned:
                    cleaned.append(t)
            meta["tags"] = cleaned
            final_path = job.job_dir / "output" / meta["file"]
            tags_path = final_path.with_name(final_path.stem + "_tags.txt")
            tags_path.write_text(" ".join(f"#{t}" for t in cleaned))

        self._persist_short_metadata(job, meta)
        return meta

    def apply_short_emphasis(self, job_id: str, index: int, words: list[str]) -> dict:
        """Re-burns this short's captions with the given keywords always
        highlighted (not just when they're the actively-spoken word),
        reusing the clip's own cached transcript and original caption style
        -- see pipeline/postedit.py for why this is fast rather than a full
        re-render."""
        job, meta = self._get_short(job_id, index)
        if job is None or meta is None:
            raise PostEditError("Short not found (job isn't done, or no such clip index).")

        transcript_path = job.job_dir / "transcript.json"
        if not transcript_path.exists():
            raise PostEditError("This job's cached transcript is missing, so captions can't be rebuilt.")
        transcript = Transcript.from_json(json.loads(transcript_path.read_text()))
        all_words = transcript.flat_words()
        clip_words = [w for w in all_words if w.start < meta["end"] and w.end > meta["start"]]
        if not clip_words:
            raise PostEditError("Couldn't find this clip's words in the cached transcript.")

        work_dir = job.job_dir / "work"
        overlaid = work_dir / f"clip{index}_02_overlaid.mp4"
        vertical = work_dir / f"clip{index}_01_vertical.mp4"
        source_for_captions = overlaid if overlaid.exists() else vertical

        final_path = job.job_dir / "output" / meta["file"]
        ass_path = work_dir / f"clip{index}.ass"
        tmp_path = final_path.with_suffix(".emphasistmp.mp4")

        emphasis_set = {w.strip().lower() for w in words if w.strip()}
        postedit.reburn_captions_with_emphasis(
            source_for_captions, ass_path, tmp_path, final_path,
            clip_words, meta["start"], emphasis_set,
            video_width=job.cfg.reframe.target_width, video_height=job.cfg.reframe.target_height,
            caption_cfg=job.cfg.captions,
        )
        meta["emphasis_words"] = sorted(emphasis_set)
        self._persist_short_metadata(job, meta)
        logger.info("Job %s short %d: re-burned captions emphasizing %s", job_id, index, sorted(emphasis_set))
        return meta

    def suggest_short_captions(self, job_id: str, index: int) -> list[dict]:
        job, meta = self._get_short(job_id, index)
        if job is None or meta is None:
            raise PostEditError("Short not found (job isn't done, or no such clip index).")
        text = meta.get("transcript_excerpt") or ""
        return suggest_captions(
            text,
            use_llm=job.cfg.highlights.use_llm,
            ollama_url=job.cfg.highlights.ollama_url,
            ollama_model=job.cfg.highlights.ollama_model,
        )
