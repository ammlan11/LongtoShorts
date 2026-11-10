"""Local, offline transcription with word-level timestamps.

Two engines, same output (a `Transcript`):

* **mlx** -- mlx-whisper, runs on the Apple-Silicon GPU. By far the fastest and
  lightest on RAM for an M-series Mac. Picked automatically (engine="auto")
  when it's installed on an arm64 Mac.
* **faster-whisper** -- CTranslate2 on CPU (or CUDA). Works everywhere. Uses
  the *batched* pipeline by default, which decodes several 30 s windows at once.

Both are run with *fast decoding* settings (greedy search, no temperature
fallback, no conditioning on previous text). The library defaults (beam 5 +
a fallback that re-decodes any low-confidence window up to 6 more times) cost
roughly 6-7x more time for a transcript that is no better for caption
purposes -- that was the main reason a 13 minute video took ~30 minutes.
See README "Why transcription used to be slow".

Nothing here calls out to a paid API. The first run for a given model
downloads open weights from Hugging Face once; after that it's fully offline.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("shorts_ai.transcribe")

_MODEL_CACHE: dict[tuple[str, str, str, int], object] = {}


def _resolve_cpu_threads(cpu_threads: int) -> int:
    """faster-whisper's engine (CTranslate2) defaults to 4 CPU threads no
    matter how many cores are actually available -- on an 8-10 core Mac
    that's leaving well over half the machine idle during the slowest stage
    of the whole pipeline. 0 (the config default) means "use every logical
    core"; anything else is used as-is so you can deliberately leave
    headroom for other apps."""
    if cpu_threads and cpu_threads > 0:
        return cpu_threads
    return os.cpu_count() or 4


@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment]

    def flat_words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]

    def to_json(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        segments = [
            Segment(
                id=s["id"],
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s["words"]],
            )
            for s in data["segments"]
        ]
        return cls(language=data["language"], duration=data["duration"], segments=segments)


def _get_model(model_size: str, device: str, compute_type: str, cpu_threads: int = 0):
    resolved_threads = _resolve_cpu_threads(cpu_threads) if device == "cpu" else 0
    key = (model_size, device, compute_type, resolved_threads)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel  # imported lazily — heavy import

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s cpu_threads=%s",
            model_size, device, compute_type, resolved_threads or "default",
        )
        kwargs = {"device": device, "compute_type": compute_type}
        if device == "cpu":
            kwargs["cpu_threads"] = resolved_threads
        _MODEL_CACHE[key] = WhisperModel(model_size, **kwargs)
    return _MODEL_CACHE[key]


# mlx-whisper model repos on Hugging Face (mlx-community) for the usual sizes.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def _mlx_available() -> bool:
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("mlx_whisper") is not None
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_engine(engine: str, device: str = "cpu") -> str:
    """Maps the config value (auto | mlx | faster-whisper) to a concrete engine.
    "auto" prefers MLX on an Apple-Silicon Mac when mlx-whisper is installed;
    everything else (Windows, Linux, Intel Mac, CUDA) uses faster-whisper."""
    engine = (engine or "auto").lower().replace("_", "-")
    if engine in ("faster-whisper", "fw", "cpu"):
        return "faster-whisper"
    if engine == "mlx":
        if not _mlx_available():
            logger.warning("engine=mlx requested but mlx-whisper isn't usable here; using faster-whisper")
            return "faster-whisper"
        return "mlx"
    # auto
    if device == "cuda":
        return "faster-whisper"
    return "mlx" if _mlx_available() else "faster-whisper"


def _extract_wav(video_path: str | Path, out_wav: str | Path) -> None:
    """16 kHz mono PCM -- what Whisper wants. MLX has no decoder of its own
    beyond shelling out to ffmpeg, so we do it explicitly with the same
    ffmpeg binary the rest of the pipeline uses."""
    from .ffmpeg_util import get_ffmpeg

    cmd = [get_ffmpeg(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr[-1500:]}")


def _transcribe_mlx(video_path, model_size, language, progress_cb, mlx_repo) -> Transcript:
    import mlx_whisper  # imported lazily -- heavy, Apple-Silicon only

    repo = mlx_repo or _MLX_REPOS.get(model_size, model_size)
    logger.info("Transcribing with MLX Whisper (%s)", repo)
    with tempfile.TemporaryDirectory(prefix="shorts_ai_wav_") as tmp:
        wav = Path(tmp) / "audio.wav"
        _extract_wav(video_path, wav)
        # MLX can't report progress mid-run; mark that the slow part started.
        if progress_cb:
            progress_cb(0.0, 1.0)
        result = mlx_whisper.transcribe(
            str(wav),
            path_or_hf_repo=repo,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            verbose=None,
        )

    segments: list[Segment] = []
    for seg in result.get("segments", []):
        words = [
            Word(word=w["word"].strip(), start=float(w["start"]), end=float(w["end"]),
                 prob=float(w.get("probability", 1.0)))
            for w in seg.get("words", [])
            if w["word"].strip()
        ]
        text = seg["text"].strip()
        if not text and not words:
            continue
        segments.append(Segment(id=len(segments), start=float(seg["start"]), end=float(seg["end"]),
                                text=text, words=words))
    duration = float(segments[-1].end) if segments else 0.0
    try:
        from .ffmpeg_util import get_ffprobe

        out = subprocess.run(
            [get_ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True,
        )
        duration = float(out.stdout.strip()) or duration
    except Exception:  # pragma: no cover - duration is cosmetic here
        pass
    if progress_cb:
        progress_cb(duration, duration)
    return Transcript(language=result.get("language") or (language or "en"), duration=duration, segments=segments)


def _transcribe_faster_whisper(video_path, model_size, device, compute_type, language,
                               progress_cb, cpu_threads, batched, batch_size) -> Transcript:
    model = _get_model(model_size, device, compute_type, cpu_threads)

    # Fast decoding -- see module docstring. beam_size=1 (greedy),
    # temperature=0.0 (a single value disables the 6-step fallback re-decode),
    # condition_on_previous_text=False (also stops one bad window poisoning
    # the next and removes a sequential dependency).
    common = dict(
        language=language,
        word_timestamps=True,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    if batched:
        from faster_whisper import BatchedInferencePipeline

        pipeline = BatchedInferencePipeline(model=model)
        segments_iter, info = pipeline.transcribe(str(video_path), batch_size=batch_size, **common)
    else:
        segments_iter, info = model.transcribe(str(video_path), **common)

    segments: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(word=w.word.strip(), start=w.start, end=w.end, prob=w.probability)
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(Segment(id=len(segments), start=seg.start, end=seg.end, text=seg.text.strip(), words=words))
        if progress_cb:
            progress_cb(seg.end, info.duration)
    return Transcript(language=info.language, duration=info.duration, segments=segments)


def transcribe_video(
    video_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
    progress_cb=None,
    cpu_threads: int = 0,
    engine: str = "auto",
    batched: bool = True,
    batch_size: int = 8,
    mlx_repo: Optional[str] = None,
) -> Transcript:
    """Runs Whisper on the audio track of `video_path` and returns a Transcript
    with word-level timestamps.

    `engine`: "auto" (MLX on Apple Silicon if installed, else faster-whisper),
    "mlx", or "faster-whisper". `batched`/`batch_size` apply to faster-whisper.

    If `cache_path` exists, the cached transcript is loaded instead of
    re-running the model (transcription is the slowest stage, so re-running a
    pipeline on the same upload should be instant).
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        logger.info("Using cached transcript at %s", cache_path)
        return Transcript.from_json(json.loads(cache_path.read_text()))

    chosen = resolve_engine(engine, device)
    transcript: Optional[Transcript] = None
    if chosen == "mlx":
        try:
            transcript = _transcribe_mlx(video_path, model_size, language, progress_cb, mlx_repo)
        except Exception as exc:  # e.g. model download blocked/offline -- don't fail the job
            logger.warning("MLX transcription failed (%s); falling back to faster-whisper", exc)
    if transcript is None:
        transcript = _transcribe_faster_whisper(
            video_path, model_size, device, compute_type, language,
            progress_cb, cpu_threads, batched, batch_size,
        )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(transcript.to_json(), indent=2))

    return transcript
