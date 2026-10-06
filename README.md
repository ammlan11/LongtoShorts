# Shorts AI

Turn a long-form video into several YouTube-Shorts-ready vertical clips:
auto-detected highlight moments, smart 9:16 (or square / 4:5) reframing
that follows the speaker's face, and animated word-by-word captions in a
style you pick. It's a small local web app — you run it on your own
machine, open a browser tab, drop in a video (or paste a link), **review a
shortlist of candidate moments and tweak them**, render just the ones you
want, then touch them up (trim, keywords, AI-suggested captions) on the
results page.

Everything runs locally. Transcription uses an open-source Whisper model
(via `faster-whisper`) that downloads once and then runs fully offline.
Highlight selection and caption suggestions use plain heuristics by
default; if you install [Ollama](https://ollama.com) and pull a model, the
tool uses it to sharpen highlight picks, titles and caption suggestions.
Nothing here calls a paid API or needs an API key.

## How it works: analyze, review, render, edit

1. **Analyze** — download (if a link) and transcribe the video (local
   Whisper), then score a shortlist of candidate moments (more than you'll
   likely use, typically 5-8). This is the slow step and only happens once
   per video. Per-stage timings (download / transcription / highlight
   picking) are shown on the review screen.
2. **Review & render** — every candidate shows its time range, suggested
   title and the plain-language reasons it was picked. Uncheck the ones you
   don't want, nudge start/end, retitle, pick a caption style and aspect
   ratio, then render. Only confirmed clips get the cut → reframe → caption
   burn-in pass.
3. **Edit (results page)** — each rendered short has an *Edit* panel:
   - **Trim** — cut seconds off the start/end of the already-rendered file.
     Fast, but captions near the new edge are not re-timed.
   - **Title & keywords** — saved as upload-ready hashtags in a `_tags.txt`
     file next to the clip.
   - **Emphasize keywords** — chosen words always get the highlighted
     caption treatment; re-burns captions from cached intermediates (a few
     seconds, no full re-render).
   - **AI caption suggestions** — 3 title + hashtag options from the clip's
     own transcript, via local Ollama if running, otherwise a built-in
     heuristic fallback.

## What the pipeline does, per clip

1. **Transcribe** — word-level timestamps, cached per job.
2. **Find highlights** — scores sliding windows of the transcript for hook
   strength, concrete numbers, pacing and engagement language; snaps
   boundaries to sentence starts/ends. Optional Ollama refinement. Also:
   - **Most replayed** — for YouTube links, uses YouTube's "most replayed"
     heatmap to tag (and favor) the most-rewatched window.
   - **Recap/preview detection** — flags intros that just preview the whole
     video (phrase match + topic-overlap heuristic); they're penalized and
     unchecked by default, but still shown.
3. **Reframe** — OpenCV face detection with a smoothed, linearly
   interpolated pan (no jerky step-pans); steady center-crop fallback when
   no face is found.
4. **Captions** — word-by-word animated captions burned in via ffmpeg
   (libass), using a bundled font so rendering is identical on every OS.

Stat-card / chart overlays (`app/pipeline/overlays.py`) are **retired**:
they were low-contrast and misfired on casual phrasing like "a hundred
bucks". The code is kept, but `overlays.enabled` is `false` and the UI/API
no longer expose it.

## Caption styles & aspect ratios

Caption presets live in `app/pipeline/presets.py`: **Bold highlight**,
**Clean white**, **Boxed**, **Minimal top**, **Neon pop**, plus a **Custom**
option in the UI (position, size, text/highlight colors, background box).
Aspect presets: **9:16**, **1:1**, **4:5**.

## Requirements

- Python 3.10-3.12 (3.12 is the tested version — see note below)
- [ffmpeg](https://ffmpeg.org/) on your PATH, built with subtitle (libass)
  support. On macOS, Homebrew's default `ffmpeg` formula does **not**
  include it — run `brew install ffmpeg-full` (the app auto-detects it).
  `setup.sh` checks and warns.
- RAM: 8GB works but is tight (see Performance notes). 16GB is comfortable.
- Optional: [Ollama](https://ollama.com/download) with a model pulled
  (e.g. `ollama pull llama3.1`) for sharper picks/captions.

No GPU required; everything defaults to CPU.

**Why the Python version matters:** `faster-whisper` depends on PyAV, which
needs prebuilt wheels for your exact Python version. On brand-new Pythons
(3.13+) those may not exist, forcing a source build that can fail with
cryptic C errors. `setup.sh` checks this for you.

## Setup

```bash
git clone <this repo>
cd shorts-ai
./setup.sh
```

Or by hand:

```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First transcription downloads the Whisper weights from Hugging Face (a few
hundred MB for the default `small` model); after that it's offline.

## Run it

```bash
source .venv/bin/activate
python run.py
```

Open **http://localhost:8000**. Only paste links to video you have the right
to download and repurpose.

## Configuration

Defaults live in `config.yaml` (each option commented inline); caption and
aspect presets live in `app/pipeline/presets.py`. Worth knowing:

- `transcription.model_size`: `tiny` (fastest, least accurate) → `small`
  (default) → `medium`/`large-v3` (more accurate, much slower on CPU).
- `transcription.cpu_threads`: defaults to `4`. `0` uses every core, which is
  fastest on a machine with a fan but runs hot on a fanless laptop (e.g. a
  MacBook Air) and can end up slower once the chip throttles.
- `highlights.use_llm`: if Ollama isn't reachable the tool silently falls
  back to heuristics — nothing breaks.
- `reframe.smoothing_window`: higher = steadier pans, lower = more reactive.
- `overlays.enabled`: `false` (feature retired, see above).

## Performance notes

Transcription dominates runtime. On an 8GB machine, Whisper + ffmpeg +
a browser can push the system into heavy swapping, which makes everything
slow *and* hot. If runs feel slow: close other apps (especially browser
tabs), keep `cpu_threads` modest, or try `model_size: tiny`. Keep
`yt-dlp` updated (`pip install -U yt-dlp`) — stale versions are the usual
cause of slow or failing link downloads. There is no hard cap on source
video length; 44-60 minute sources are supported (candidate generation is
linear-time), but expect transcription time to scale with length.

## Project layout

```
shorts-ai/
├── setup.sh                 preflight + venv + install (recommended)
├── run.py                   entrypoint (starts the web server)
├── config.yaml              all tunable settings
├── app/
│   ├── main.py              FastAPI routes (upload, presets, render, poll,
│   │                        download, and the /shorts/{i}/trim|update|
│   │                        emphasis|suggest-captions edit endpoints)
│   ├── jobs.py              background job runner: analyze -> awaiting_review
│   │                        -> render -> done, plus post-render edit methods
│   ├── config.py            config loading
│   ├── assets/fonts/        bundled DejaVu Sans (regular + bold)
│   ├── static/              single-page frontend (upload -> review -> results/edit)
│   └── pipeline/
│       ├── transcribe.py        local Whisper wrapper (cached transcripts)
│       ├── highlights.py        heuristic + optional-LLM scoring, recap detection
│       ├── retention.py         YouTube "most replayed" window detection
│       ├── presets.py           caption style + aspect ratio presets
│       ├── reframe.py           face-tracked, smoothed crop
│       ├── captions.py          word-level animated .ass captions (+ emphasis)
│       ├── postedit.py          fast post-render trim / caption re-burn
│       ├── caption_suggest.py   AI (Ollama) / heuristic title+hashtag suggestions
│       ├── overlays.py          retired stat-card overlays (disabled)
│       ├── palette.py           colors used by the overlay graphics
│       ├── llm.py               optional local Ollama client
│       ├── youtube.py           yt-dlp wrapper for the "paste a link" path
│       ├── ffmpeg_util.py       finds an ffmpeg with libass support
│       └── render.py            analyze_video() + render_selected()
└── scripts/                 smoke/regression tests + synthetic test-video generator
```

## Testing

No model download or network is needed for these; they use a hand-built
transcript. Generate the synthetic source clip first:

```bash
python scripts/make_test_video.py /tmp/shorts_ai_test/source.mp4
python scripts/test_pipeline_offline.py     # analyze + render directly
python scripts/test_api_e2e.py              # full HTTP flow via TestClient
python scripts/test_postedit.py             # trim / tags / emphasis / suggestions
python scripts/test_postedit_api.py         # same, through the HTTP endpoints
python scripts/test_reframe_long_clip.py    # regression: 58s clip reframe
```

(`test_pipeline.py` is the original live-Whisper variant and does need the
model download.)

## Known limitations

- Face tracking uses a classic Haar-cascade detector: reliable for a
  front-facing talking head, weaker for side angles, screen-share footage or
  multiple people (falls back to a steady center-crop).
- Trimming a rendered short cuts the final file directly, so captions can end
  abruptly at the new edge; a full re-render would fix that but takes as long
  as the original render.
- Post-render caption emphasis needs the job's cached intermediates; if a
  job folder under `storage/` was cleaned up, re-render instead.
- This is a single-user local tool: in-memory job state (lost on restart), no
  auth, and the server binds to `0.0.0.0` by default — change
  `server.host` to `127.0.0.1` if you don't want it reachable on your LAN.

## Third-party assets

`app/assets/fonts/DejaVuSans*.ttf` are DejaVu fonts, distributed under the
DejaVu/Bitstream Vera free license.
