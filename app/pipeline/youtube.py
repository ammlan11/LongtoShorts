"""Downloads a video from a URL (YouTube and anything else yt-dlp supports)
so it can run through the same local pipeline as an uploaded file.

Uses yt-dlp — the actively maintained, open-source youtube-dl fork — purely
as a local download step; nothing here calls a paid API. Only point this at
video you actually have the right to download and repurpose (your own
uploads, or anything whose license/terms allow it) — this tool doesn't and
can't check that for you, that judgment call stays with you.

A note on speed: YouTube regularly changes the signature/throttling scheme
its CDN uses, and yt-dlp ships near-weekly releases specifically to keep up
with that. An yt-dlp install that's gone stale for a few months is the #1
cause of a "YouTube download" that crawls at a tiny fraction of your real
bandwidth -- if downloads feel slow, `pip install -U yt-dlp` first, before
assuming it's this app or your connection.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("shorts_ai.youtube")

ProgressCB = Optional[Callable[[float], None]]  # fraction 0.0-1.0


class DownloadError(RuntimeError):
    pass


@dataclass
class DownloadResult:
    path: Path
    # YouTube's own "most replayed" retention graph (the gray heatmap under
    # the scrubber), as yt-dlp extracts it: a list of
    # {"start_time": float, "end_time": float, "value": float 0-1} segments
    # covering the whole video. None when yt-dlp didn't return one — not
    # every video has enough view history for YouTube to compute it, and
    # non-YouTube sources never have it.
    heatmap: Optional[list[dict]] = None
    download_seconds: float = 0.0


def download_video(
    url: str, dest_dir: Path, max_height: int = 1080, progress_cb: ProgressCB = None
) -> DownloadResult:
    """Downloads `url` into `dest_dir` as an mp4 and returns the file path
    plus YouTube's "most replayed" heatmap data when available. Video is
    capped at `max_height` (1080p by default) — plenty for a Shorts source
    and much faster than pulling a 4K master.

    The format selector deliberately does NOT restrict to ext=mp4 streams:
    every clip gets fully re-encoded with libx264 in the very first pipeline
    stage anyway (render.py's _cut_clip), so the source container/codec
    doesn't need to already be mp4/h264 — restricting to it just risked
    ruling out the fastest-to-fetch format for no benefit. Fragments are
    also downloaded concurrently, which meaningfully speeds up the common
    case of a DASH-fragmented stream.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError(
            "yt-dlp isn't installed. Run: pip install -r requirements.txt"
        ) from e

    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")

    def _hook(d: dict) -> None:
        if not progress_cb:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes")
            if total and downloaded is not None:
                try:
                    progress_cb(min(0.99, downloaded / total))
                except Exception:  # noqa: BLE001 — progress reporting must never break the download
                    pass
        elif d.get("status") == "finished":
            try:
                progress_cb(1.0)
            except Exception:  # noqa: BLE001
                pass

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": (
            f"bestvideo[height<={max_height}]+bestaudio"
            f"/best[height<={max_height}]/best"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        # Fetch multiple fragments of a DASH stream in parallel instead of
        # one at a time -- often the single biggest download-speed win for
        # a typical YouTube video, independent of raw connection speed.
        "concurrent_fragment_downloads": 4,
        "retries": 5,
        "fragment_retries": 5,
        "progress_hooks": [_hook],
    }

    started = time.perf_counter()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info, outtmpl=outtmpl))
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Couldn't download that video: {e}") from e
    elapsed = time.perf_counter() - started

    # merge_output_format can change the final extension after post-processing
    if not path.exists():
        candidate = path.with_suffix(".mp4")
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise DownloadError("Download reported success but the output file is missing.")

    heatmap = info.get("heatmap") or None
    logger.info(
        "Downloaded %s -> %s in %.1fs (heatmap: %s)",
        url, path, elapsed, "yes" if heatmap else "none",
    )
    return DownloadResult(path=path, heatmap=heatmap, download_seconds=elapsed)
