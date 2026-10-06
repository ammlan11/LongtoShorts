"""Local, offline transcription with word-level timestamps via faster-whisper.

Nothing here calls out to a paid API — the model runs on your own CPU/GPU.
The first run for a given model_size downloads open weights from Hugging Face
once; after that it's fully offline.
"""
from __future__ import annotations

import json
import logging
import os
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


def transcribe_video(
    video_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
    progress_cb=None,
    cpu_threads: int = 0,
) -> Transcript:
    """Runs Whisper on the audio track of `video_path` and returns a Transcript
    with word-level timestamps. ffmpeg (via faster-whisper's own decoder) pulls
    the audio directly from the video container — no separate extraction step
    needed.

    If `cache_path` exists, the cached transcript is loaded instead of
    re-running the model (transcription is the slowest stage, so re-running a
    pipeline on the same upload during development should be instant).

    `cpu_threads` (0 = auto, use every core) is the single biggest free
    speed-up available here — see _resolve_cpu_threads() above.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        logger.info("Using cached transcript at %s", cache_path)
        return Transcript.from_json(json.loads(cache_path.read_text()))

    model = _get_model(model_size, device, compute_type, cpu_threads)

    segments_iter, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[Segment] = []
    for i, seg in enumerate(segments_iter):
        words = [
            Word(word=w.word.strip(), start=w.start, end=w.end, prob=w.probability)
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(Segment(id=i, start=seg.start, end=seg.end, text=seg.text.strip(), words=words))
        if progress_cb:
            progress_cb(seg.end, info.duration)

    transcript = Transcript(language=info.language, duration=info.duration, segments=segments)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(transcript.to_json(), indent=2))

    return transcript
