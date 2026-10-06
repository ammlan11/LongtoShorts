"""Resolves which ffmpeg/ffprobe binaries the whole pipeline uses.

This exists because of a real, easy-to-hit gap: Homebrew's default `ffmpeg`
formula on macOS ships WITHOUT libass support, so the `ass` filter (used by
captions.py to burn in subtitles) is simply missing — `ffmpeg -h filter=ass`
reports "Unknown filter 'ass'" and every render with captions enabled fails.
`ffmpeg-full` (also in homebrew-core, no extra tap needed) includes libass,
but is keg-only, so it's never the `ffmpeg` found on PATH by default.

Rather than have captions.py silently fail while every other stage works,
we resolve ONE ffmpeg binary for the whole pipeline (preferring PATH, falling
back to ffmpeg-full's keg-only location if PATH's build lacks libass) and
use that same binary everywhere — so every stage of a given run agrees on
the same ffmpeg build instead of mixing two different ones.
"""
from __future__ import annotations

import functools
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("shorts_ai.ffmpeg")

# Homebrew keg-only install locations for ffmpeg-full, by Mac architecture.
_CANDIDATE_KEGONLY_PATHS = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",  # Apple Silicon
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",     # Intel
]

_INSTALL_HINT = (
    "No ffmpeg with subtitle (libass) support was found, so captions can't "
    "be burned in. Your ffmpeg works for everything else. Fix: "
    "`brew install ffmpeg-full` (macOS, no extra tap needed — it installs "
    "alongside your existing ffmpeg without replacing it), or on Linux, "
    "install/build ffmpeg with --enable-libass. You can also disable "
    "captions for now (captions.enabled: false in config.yaml, or uncheck "
    "'Auto captions' in the upload form)."
)


def _supports_ass(ffmpeg_bin: str) -> bool:
    try:
        r = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-h", "filter=ass"],
            capture_output=True, text=True, timeout=10,
        )
        return "Unknown filter" not in r.stdout and "Unknown filter" not in r.stderr
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def get_ffmpeg(require_ass: bool = False) -> str:
    """Returns the ffmpeg binary the pipeline should use. Resolved once per
    process and cached. Pass require_ass=True (captions.py does) to raise a
    clear, actionable error immediately if no libass-capable build exists,
    instead of letting ffmpeg fail deep inside a subprocess call."""
    on_path = shutil.which("ffmpeg") or "ffmpeg"
    on_path_has_ass = _supports_ass(on_path)

    if on_path_has_ass:
        return on_path

    for candidate in _CANDIDATE_KEGONLY_PATHS:
        if Path(candidate).exists() and _supports_ass(candidate):
            logger.info("Default ffmpeg (%s) lacks libass; using %s instead", on_path, candidate)
            return candidate

    if require_ass:
        raise RuntimeError(_INSTALL_HINT)

    # Captions aren't needed for this call — fine to use the PATH binary
    # even without libass (cutting/reframing/overlays don't need it).
    return on_path


@functools.lru_cache(maxsize=1)
def get_ffprobe() -> str:
    """ffprobe never needs libass, but resolved here too so every pipeline
    module gets its binary path from one place."""
    return shutil.which("ffprobe") or "ffprobe"
