"""Loads config.yaml into plain dicts/dataclasses. Deliberately simple —
this is a single-user local tool, not a multi-tenant service."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TranscriptionConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    # faster-whisper's underlying engine (CTranslate2) defaults to just 4 CPU
    # threads regardless of how many cores your machine actually has, which
    # left most of a multi-core machine idle during the slowest pipeline
    # stage. 0 means "auto" -- use every logical core (see transcribe.py's
    # _resolve_cpu_threads()), which is the fastest setting *if* your machine
    # has active cooling. On a fanless machine (e.g. a MacBook Air) pinning
    # every core -- including the weaker efficiency cores, which barely help
    # this workload -- generates far more heat than it's worth and can
    # eventually trigger thermal throttling, which then makes things slow
    # again anyway. 4 is a safer default that sticks to roughly a
    # performance-core's worth of parallelism; raise it (or set 0) if your
    # machine has a fan and you want to trade heat for a bit more speed.
    cpu_threads: int = 4


@dataclass
class HighlightsConfig:
    clip_count: int = 3
    min_duration_sec: float = 20
    max_duration_sec: float = 59
    use_llm: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"


@dataclass
class ReframeConfig:
    target_aspect: str = "9:16"
    target_width: int = 1080
    target_height: int = 1920
    smoothing_window: int = 15


@dataclass
class CaptionsConfig:
    enabled: bool = True
    font: str = "DejaVu Sans"
    font_size: int = 64
    highlight_color: str = "&H0026D9D9"
    base_color: str = "&H00FFFFFF"
    outline_color: str = "&H00161616"
    back_color: str = "&H64000000"
    position: str = "lower_third"
    border_style: int = 1
    outline_width: int = 5
    shadow_width: int = 2
    # Which entry in pipeline/presets.CAPTION_PRESETS produced the fields
    # above, if any — kept around just so the UI can show what's selected.
    preset: Optional[str] = None


@dataclass
class OverlaysConfig:
    # Off by default: the generated stat-card graphics (a number pulled from
    # the transcript + a short label, in a translucent card) turned out to
    # be low-contrast and hard to read against real footage, and the number
    # extraction also fires on casual phrasing ("a hundred bucks") that
    # isn't really a stat worth calling out. Kept in the codebase (rather
    # than deleted) in case it's worth revisiting with a redesigned,
    # higher-contrast card, but no longer exposed in the UI -- see
    # app/main.py and app/static/index.html.
    enabled: bool = False
    min_stats_for_card: int = 1
    min_stats_for_chart: int = 2
    card_seconds: float = 3.2
    theme: str = "dark"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    storage_dir: Path
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    highlights: HighlightsConfig = field(default_factory=HighlightsConfig)
    reframe: ReframeConfig = field(default_factory=ReframeConfig)
    captions: CaptionsConfig = field(default_factory=CaptionsConfig)
    overlays: OverlaysConfig = field(default_factory=OverlaysConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def uploads_dir(self) -> Path:
        d = self.storage_dir / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def jobs_dir(self) -> Path:
        d = self.storage_dir / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _dc(cls, data: dict[str, Any] | None):
    data = data or {}
    valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**valid)


def load_config(path: str | Path | None = None) -> AppConfig:
    # NOTE: ROOT is the project root (shorts-ai/), i.e. config.py's
    # grandparent directory. Both paths below must resolve relative to ROOT,
    # not ROOT.parent (one level too high, e.g. the user's home directory) —
    # that off-by-one previously meant config.yaml was silently never found
    # (defaults always applied regardless of edits) and job storage landed
    # outside the project folder entirely.
    path = Path(path) if path else ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    storage_dir = ROOT / raw.get("paths", {}).get("storage_dir", "storage")
    return AppConfig(
        storage_dir=storage_dir,
        transcription=_dc(TranscriptionConfig, raw.get("transcription")),
        highlights=_dc(HighlightsConfig, raw.get("highlights")),
        reframe=_dc(ReframeConfig, raw.get("reframe")),
        captions=_dc(CaptionsConfig, raw.get("captions")),
        overlays=_dc(OverlaysConfig, raw.get("overlays")),
        server=_dc(ServerConfig, raw.get("server")),
    )
