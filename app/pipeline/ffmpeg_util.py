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
import glob
import logging
import os
import re
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


def _extra_ffmpeg_locations() -> list[str]:
    """Places ffmpeg commonly lives when it's installed but NOT on the PATH of
    the terminal the app was started from (very common on Windows right after
    `winget install`, before the terminal is reopened). Also honours an
    explicit SHORTS_AI_FFMPEG env var (full path to ffmpeg, or its folder)."""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found: list[str] = []
    override = os.environ.get("SHORTS_AI_FFMPEG")
    if override:
        o = Path(override)
        found.append(str(o / exe) if o.is_dir() else str(o))
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        prog = os.environ.get("ProgramFiles", r"C:\Program Files")
        progdata = os.environ.get("ProgramData", r"C:\ProgramData")
        home = os.path.expanduser("~")
        patterns = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(prog, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(progdata, "chocolatey", "bin", "ffmpeg.exe"),
            os.path.join(home, "scoop", "shims", "ffmpeg.exe"),
            os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
            os.path.join(local, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "bin", "ffmpeg.exe"),
        ]
        for pat in patterns:
            found.extend(glob.glob(pat, recursive=True))
    else:
        found.extend(["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"])
    return [f for f in found if f and Path(f).exists()]


def find_ffmpeg() -> str | None:
    """Absolute path to *some* ffmpeg (PATH first, then the usual install
    locations), or None if there isn't one on this machine."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    extra = _extra_ffmpeg_locations()
    return extra[0] if extra else None


FFMPEG_MISSING_HINT = (
    "ffmpeg isn't installed (or isn't on this terminal's PATH), and Shorts AI needs it to download, "
    "cut and caption video. Install it, then CLOSE and REOPEN your terminal and start Shorts AI again:\n"
    "  Windows: winget install Gyan.FFmpeg    (or: choco install ffmpeg)\n"
    "  macOS:   brew install ffmpeg-full\n"
    "  Linux:   sudo apt install ffmpeg\n"
    "If it is installed somewhere unusual, set the SHORTS_AI_FFMPEG environment variable to its path."
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
    on_path = find_ffmpeg() or "ffmpeg"
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
    # Prefer the ffprobe that sits next to the ffmpeg we resolved (they ship
    # together), so a non-PATH ffmpeg install still gets a working ffprobe.
    found = shutil.which("ffprobe")
    if found:
        return found
    ff = find_ffmpeg()
    if ff:
        sibling = Path(ff).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if sibling.exists():
            return str(sibling)
    return "ffprobe"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Removes terminal colour codes (yt-dlp/ffmpeg error text) so messages
    shown in the web UI aren't littered with '[0;31m' junk."""
    return _ANSI_RE.sub("", text or "")
