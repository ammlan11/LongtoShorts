#!/usr/bin/env bash
# Preflight + setup for Shorts AI.
#
# This exists because the #1 source of setup pain for this project has been
# silent, confusing dependency build failures on whatever Python happens to
# be `python3` on your machine — especially a very new version (3.13/3.14)
# that doesn't have prebuilt wheels yet for faster-whisper's native
# dependencies (PyAV in particular). Rather than let pip grind through a
# doomed source build and dump a wall of C compiler errors on you, this
# script picks a known-good Python up front and tells you plainly if it
# can't find one.
#
# Usage:
#   ./setup.sh
#
# Safe to re-run — it reuses an existing .venv if one is already set up with
# a good Python version.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
info()  { echo -e "${GREEN}==>${RESET} $1"; }
warn()  { echo -e "${YELLOW}==> warning:${RESET} $1"; }
fail()  { echo -e "${RED}==> error:${RESET} $1"; exit 1; }

# ---- 1. ffmpeg ------------------------------------------------------------

if ! command -v ffmpeg >/dev/null 2>&1; then
  fail "ffmpeg is not on your PATH. Install it first:
    macOS:  brew install ffmpeg
    Ubuntu/Debian:  sudo apt-get install ffmpeg
  Then re-run ./setup.sh"
fi
info "ffmpeg found: $(ffmpeg -version | head -n1)"

# Captions need the `ass` filter, which requires ffmpeg to be built with
# libass. Homebrew's default `ffmpeg` formula on macOS does NOT include it
# (it moved to the separate, keg-only `ffmpeg-full` formula) -- so a plain
# `brew install ffmpeg` builds fine but silently can't burn in captions. The
# app auto-detects and falls back to ffmpeg-full if it's installed, so this
# is just a heads-up, not a hard failure.
if ffmpeg -hide_banner -h filter=ass 2>&1 | grep -qi "unknown filter"; then
  if [ -x /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg ] || [ -x /usr/local/opt/ffmpeg-full/bin/ffmpeg ]; then
    info "ffmpeg-full (with caption support) found alongside your default ffmpeg -- will use it for captions."
  else
    warn "Your ffmpeg lacks subtitle (libass) support, so caption burn-in will fail. Fix with:
    macOS:  brew install ffmpeg-full   (installs alongside your existing ffmpeg, no extra tap)
    Ubuntu/Debian:  rebuild ffmpeg with --enable-libass, or install a libass-enabled build
  Or disable captions in config.yaml (captions.enabled: false) if you don't need them."
  fi
else
  info "ffmpeg has caption (libass) support."
fi

# ---- 2. pick a Python that actually has prebuilt wheels for everything ----
#
# faster-whisper depends on PyAV, which needs a source build (requiring
# pkg-config + a matching system ffmpeg's dev headers) on any Python version
# too new to have a prebuilt wheel yet. 3.10-3.12 are the safe, tested range;
# 3.13+ is where this has repeatedly broken in practice.

GOOD_PY=""
for candidate in python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    GOOD_PY="$candidate"
    break
  fi
done

if [ -z "$GOOD_PY" ]; then
  if command -v python3 >/dev/null 2>&1; then
    ver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="$(echo "$ver" | cut -d. -f1)"
    minor="$(echo "$ver" | cut -d. -f2)"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
      GOOD_PY="python3"
    else
      warn "Only found python3 at version $ver. Versions 3.13+ have repeatedly
  hit source-build failures here (PyAV / faster-whisper's native deps lack
  prebuilt wheels on very new Pythons yet). Recommended fix:
    macOS:  brew install python@3.12
    Ubuntu/Debian:  sudo apt-get install python3.12 python3.12-venv
  Continuing anyway with python3 ($ver) since nothing better was found —
  if the pip install below fails with C compiler errors, that's why."
      GOOD_PY="python3"
    fi
  else
    fail "No python3 found on PATH at all. Install Python 3.12 and re-run."
  fi
fi

PY_VERSION="$("$GOOD_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
info "Using $GOOD_PY (Python $PY_VERSION)"

# ---- 3. pkg-config (only strictly needed if PyAV ends up building from source)

if ! command -v pkg-config >/dev/null 2>&1; then
  warn "pkg-config not found. This is only needed if pip has to build PyAV
  from source (no prebuilt wheel for your platform/Python combo). If the
  install below fails with 'pkg-config is required for building PyAV':
    macOS:  brew install pkg-config
    Ubuntu/Debian:  sudo apt-get install pkg-config"
fi

# ---- 4. venv ---------------------------------------------------------------

if [ -d .venv ]; then
  existing_ver="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")"
  info "Found existing .venv (Python $existing_ver)"
  if [ "$existing_ver" != "$($GOOD_PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')" ]; then
    warn ".venv is a different Python version than $GOOD_PY. If you hit
  install problems, delete .venv and re-run: rm -rf .venv && ./setup.sh"
  fi
else
  info "Creating virtual environment in .venv ..."
  "$GOOD_PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Upgrading pip ..."
pip install --upgrade pip --quiet

info "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

echo
info "Setup complete."
echo "  Next steps:"
echo "    source .venv/bin/activate"
echo "    python run.py"
echo "  Then open http://localhost:8000"
echo
echo "  The first time you transcribe a video, faster-whisper downloads the"
echo "  Whisper model weights (a few hundred MB for the default 'small' size)"
echo "  — after that, transcription is fully offline."
