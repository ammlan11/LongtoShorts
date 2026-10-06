"""Orchestrates the pipeline in two phases:

  analyze_video()   transcribe -> find candidate highlights (more than you'll
                     necessarily use) -> hand back to the caller for review.
  render_selected()  take the (possibly user-edited) selection of candidates
                     and actually cut -> reframe -> stat overlays -> captions
                     -> final short, for just those clips.

Splitting it this way is what makes the "review and edit before rendering"
step possible: the expensive transcription only happens once, and the user
decides which candidates are worth spending render time on — and can adjust
their boundaries/titles first.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

from . import captions, overlays, reframe
from .ffmpeg_util import get_ffmpeg
from .highlights import Candidate, find_highlights
from .retention import find_most_replayed_window
from .transcribe import Transcript, Word, transcribe_video

logger = logging.getLogger("shorts_ai.render")

ProgressCB = Optional[Callable[[str, float], None]]


def _cut_clip(source: Path, start: float, end: float, out_path: Path) -> None:
    duration = end - start
    fast_seek = max(0.0, start - 2.0)
    remainder = start - fast_seek
    cmd = [
        get_ffmpeg(), "-y",
        "-ss", f"{fast_seek:.3f}", "-i", str(source),
        "-ss", f"{remainder:.3f}", "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def render_clip(
    source_video: Path,
    candidate: Candidate,
    out_dir: Path,
    work_dir: Path,
    index: int,
    cfg,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_path = work_dir / f"clip{index}_00_raw.mp4"
    vertical_path = work_dir / f"clip{index}_01_vertical.mp4"
    overlaid_path = work_dir / f"clip{index}_02_overlaid.mp4"
    final_path = out_dir / f"short_{index + 1}.mp4"

    _cut_clip(source_video, candidate.start, candidate.end, raw_path)

    crop_plan = reframe.apply_reframe(
        raw_path, vertical_path,
        target_width=cfg.reframe.target_width,
        target_height=cfg.reframe.target_height,
        smoothing_window=cfg.reframe.smoothing_window,
    )

    if cfg.overlays.enabled:
        overlay_work = work_dir / f"clip{index}_overlays"
        items = overlays.build_overlay_items(
            candidate.words, candidate.start, candidate.duration, overlay_work,
            theme=cfg.overlays.theme,
            min_stats_for_card=cfg.overlays.min_stats_for_card,
            min_stats_for_chart=cfg.overlays.min_stats_for_chart,
            card_seconds=cfg.overlays.card_seconds,
            target_width=cfg.reframe.target_width,
        )
        overlays.composite_overlays(vertical_path, items, overlaid_path)
        stat_count = len(items)
    else:
        overlaid_path = vertical_path
        stat_count = 0

    if cfg.captions.enabled:
        ass_content = captions.build_ass(
            candidate.words, candidate.start,
            video_width=cfg.reframe.target_width, video_height=cfg.reframe.target_height,
            font=cfg.captions.font, font_size=cfg.captions.font_size,
            highlight_color=cfg.captions.highlight_color, base_color=cfg.captions.base_color,
            outline_color=cfg.captions.outline_color, back_color=cfg.captions.back_color,
            position=cfg.captions.position, border_style=cfg.captions.border_style,
            outline_width=cfg.captions.outline_width, shadow_width=cfg.captions.shadow_width,
        )
        ass_path = work_dir / f"clip{index}.ass"
        captions.write_ass_file(ass_content, ass_path)
        captions.burn_subtitles(overlaid_path, ass_path, final_path)
    else:
        subprocess.run([get_ffmpeg(), "-y", "-i", str(overlaid_path), "-c", "copy", str(final_path)],
                        capture_output=True, text=True, check=True)

    metadata = {
        "index": index,
        "file": final_path.name,
        "title": candidate.title,
        "reasons": candidate.reasons,
        "score": round(candidate.score, 2),
        "start": round(candidate.start, 2),
        "end": round(candidate.end, 2),
        "duration": round(candidate.duration, 2),
        "transcript_excerpt": candidate.text,
        "face_tracked": crop_plan.get("found_face", False),
        "overlay_count": stat_count,
        "most_replayed": candidate.most_replayed,
        "is_recap": candidate.is_recap,
        # Filled in later from the results page (see app/pipeline/postedit.py
        # and the /shorts/{index}/update and /emphasis endpoints) -- present
        # here from the start so the API response shape is consistent
        # whether or not either has been used yet.
        "tags": [],
        "emphasis_words": [],
    }
    (out_dir / f"short_{index + 1}.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _transcript_cache_path(job_dir: Path) -> Path:
    return job_dir / "transcript.json"


def analyze_video(
    video_path: str | Path,
    job_dir: Path,
    cfg,
    progress_cb: ProgressCB = None,
    max_candidates: Optional[int] = None,
    heatmap: Optional[list[dict]] = None,
) -> dict:
    """Phase 1: transcribe the source once (cached to job_dir/transcript.json
    so render_selected() below never has to redo it), then surface MORE
    candidate highlights than cfg.highlights.clip_count calls for — the whole
    point is to hand the caller a shortlist (3-5+ options) to review and edit,
    not a final answer.

    Returns a plain-dict summary safe to JSON-serialize straight into an API
    response: {source, language, source_duration, candidates: [...]}. Each
    candidate dict carries an `id` (its index into this list) that the caller
    echoes back — possibly with edited start/end/title — in `selections` to
    render_selected().

    `heatmap` is YouTube's own "most replayed" retention data (see
    pipeline/retention.py), passed through from the download step when the
    source was a YouTube URL. When present, the single window it points to
    either gets folded into an existing overlapping candidate (tagging it
    `most_replayed=True` and adding a reason) or, if it doesn't meaningfully
    overlap any heuristic/LLM pick, is added as its own extra candidate --
    so "the part viewers actually rewatch most" is always surfaced even if
    our own scoring missed it.
    """
    video_path = Path(video_path)
    job_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, pct: float):
        logger.info("[%s] %.0f%%", stage, pct * 100)
        if progress_cb:
            progress_cb(stage, pct)

    report("transcribing", 0.0)
    transcript_cache = _transcript_cache_path(job_dir)

    def _t_progress(done: float, total: float):
        if total:
            report("transcribing", min(0.95, done / total))

    transcript: Transcript = transcribe_video(
        video_path,
        model_size=cfg.transcription.model_size,
        device=cfg.transcription.device,
        compute_type=cfg.transcription.compute_type,
        language=cfg.transcription.language,
        cache_path=transcript_cache,
        progress_cb=_t_progress,
        cpu_threads=cfg.transcription.cpu_threads,
    )
    report("transcribing", 1.0)

    report("finding_highlights", 0.0)
    # Ask for noticeably more candidates than the configured final count, so
    # there's an actual choice to review (the UI wants 3-5+ examples to pick
    # from, not just cfg.highlights.clip_count already-decided clips).
    want = max_candidates or max(cfg.highlights.clip_count * 2, 5)
    candidates: list[Candidate] = find_highlights(
        transcript,
        clip_count=want,
        min_duration_sec=cfg.highlights.min_duration_sec,
        max_duration_sec=cfg.highlights.max_duration_sec,
        use_llm=cfg.highlights.use_llm,
        ollama_url=cfg.highlights.ollama_url,
        ollama_model=cfg.highlights.ollama_model,
    )
    report("finding_highlights", 1.0)

    if heatmap:
        window = find_most_replayed_window(
            heatmap,
            target_duration=(cfg.highlights.min_duration_sec + cfg.highlights.max_duration_sec) / 2,
            min_duration=cfg.highlights.min_duration_sec,
            max_duration=cfg.highlights.max_duration_sec,
        )
        if window:
            overlap = _find_overlapping_candidate(candidates, window.start, window.end)
            if overlap is not None:
                overlap.most_replayed = True
                overlap.reasons = [*overlap.reasons, "YouTube's own 'most replayed' data peaks here too"]
            else:
                all_words = transcript.flat_words()
                words = _words_in_range(all_words, window.start, window.end)
                text = " ".join(w.word for w in words)
                candidates.insert(0, Candidate(
                    start=window.start,
                    end=window.end,
                    text=text,
                    words=words,
                    score=max((c.score for c in candidates), default=0.0) + 1.0,
                    reasons=["This is the single moment YouTube's own retention data shows viewers replay most"],
                    title=(text[:50] + "...") if len(text) > 50 else (text or "Most replayed moment"),
                    most_replayed=True,
                ))

    candidate_dicts = [
        {
            "id": i,
            "start": round(c.start, 2),
            "end": round(c.end, 2),
            "duration": round(c.duration, 2),
            "title": c.title or (c.text[:50] + "..." if len(c.text) > 50 else c.text),
            "score": round(c.score, 2),
            "reasons": c.reasons,
            "text": c.text,
            "most_replayed": c.most_replayed,
            "is_recap": c.is_recap,
        }
        for i, c in enumerate(candidates)
    ]

    analysis = {
        "source": video_path.name,
        "language": transcript.language,
        "source_duration": round(transcript.duration, 2),
        "candidates": candidate_dicts,
    }
    (job_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    report("done", 1.0)
    return analysis


def _words_in_range(words: list[Word], start: float, end: float) -> list[Word]:
    """Overlap test rather than strict containment — a word that straddles a
    user-dragged boundary should still show up rather than vanishing."""
    return [w for w in words if w.start < end and w.end > start]


def _find_overlapping_candidate(
    candidates: list[Candidate], start: float, end: float, min_overlap_frac: float = 0.4
) -> Optional[Candidate]:
    """Returns whichever existing candidate shares the most time with
    [start, end], as long as the overlap covers at least `min_overlap_frac`
    of the heatmap window — used to fold YouTube's "most replayed" window
    into a candidate we already picked instead of adding a near-duplicate
    clip right next to it."""
    window_dur = max(end - start, 0.001)
    best: Optional[Candidate] = None
    best_overlap = 0.0
    for c in candidates:
        overlap = max(0.0, min(end, c.end) - max(start, c.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = c
    if best is not None and (best_overlap / window_dur) >= min_overlap_frac:
        return best
    return None


def render_selected(
    video_path: str | Path,
    job_dir: Path,
    cfg,
    selections: list[dict],
    progress_cb: ProgressCB = None,
) -> dict:
    """Phase 2: take the (possibly user-edited) list of selections from
    analyze_video()'s candidates — each a dict with at least start/end, and
    optionally title/score/reasons carried over or overridden by the user —
    and actually render just those clips.

    Re-loads the transcript from the cache analyze_video() wrote, so this
    phase never re-runs the (slow) Whisper pass even if the user takes a
    while deciding what to keep. Any style/aspect-ratio choice is applied by
    the caller mutating `cfg` before calling this (see jobs.py) rather than
    threaded through here, since it's a per-job, not per-clip, choice.
    """
    video_path = Path(video_path)
    out_dir = job_dir / "output"
    work_dir = job_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, pct: float):
        logger.info("[%s] %.0f%%", stage, pct * 100)
        if progress_cb:
            progress_cb(stage, pct)

    transcript_cache = _transcript_cache_path(job_dir)
    transcript: Transcript = transcribe_video(
        video_path,
        model_size=cfg.transcription.model_size,
        device=cfg.transcription.device,
        compute_type=cfg.transcription.compute_type,
        language=cfg.transcription.language,
        cache_path=transcript_cache,
        cpu_threads=cfg.transcription.cpu_threads,
    )
    all_words = transcript.flat_words()

    candidates: list[Candidate] = []
    for sel in selections:
        start = float(sel["start"])
        end = float(sel["end"])
        words = _words_in_range(all_words, start, end)
        text = sel.get("text") or " ".join(w.word for w in words)
        candidates.append(
            Candidate(
                start=start,
                end=end,
                text=text,
                words=words,
                score=float(sel.get("score", 0.0)),
                reasons=list(sel.get("reasons", [])),
                title=sel.get("title", ""),
                most_replayed=bool(sel.get("most_replayed", False)),
                is_recap=bool(sel.get("is_recap", False)),
            )
        )

    report("rendering", 0.0)
    results = []
    n = max(len(candidates), 1)
    for i, c in enumerate(candidates):
        report("rendering", i / n)
        meta = render_clip(video_path, c, out_dir, work_dir, i, cfg)
        results.append(meta)
    report("rendering", 1.0)

    summary = {
        "source": video_path.name,
        "language": transcript.language,
        "source_duration": round(transcript.duration, 2),
        "shorts": results,
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report("done", 1.0)
    return summary
