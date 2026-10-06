"""Word-level animated captions (the "TikTok-style" bouncing-word look),
rendered as an .ass subtitle file and burned into the video with ffmpeg's
`ass` filter. Fully local — no cloud captioning API involved, since we
already have word timestamps from the Whisper transcript.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .ffmpeg_util import get_ffmpeg
from .transcribe import Word

logger = logging.getLogger("shorts_ai.captions")

_ALIGN = {"lower_third": 2, "center": 5, "top": 8}


def _ts(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _clean(word: str) -> str:
    return word.replace("{", "").replace("}", "").replace("\\", "")


def _group_words(words: list[Word], max_chunk_words: int = 4, max_chunk_sec: float = 1.6) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current:
            gap = w.start - current[-1].end
            duration = w.end - current[0].start
            if gap > 0.6 or len(current) >= max_chunk_words or duration > max_chunk_sec:
                chunks.append(current)
                current = []
        current.append(w)
        if current[-1].word.endswith((".", "?", "!")):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def build_ass(
    words: list[Word],
    clip_start: float,
    video_width: int,
    video_height: int,
    font: str = "DejaVu Sans",
    font_size: int = 84,
    highlight_color: str = "&H0026D9D9",
    base_color: str = "&H00FFFFFF",
    outline_color: str = "&H00161616",
    back_color: str = "&H64000000",
    position: str = "lower_third",
    border_style: int = 1,
    outline_width: int = 5,
    shadow_width: int = 2,
    emphasis_words: set[str] | None = None,
) -> str:
    """Returns the full text content of an .ass file. Word times are shifted
    from absolute (original video) time to clip-relative time using
    `clip_start`.

    `border_style`/`outline_width`/`shadow_width`/`back_color` are what the
    caption style presets (see presets.py) vary to get different looks —
    1 = text with an outline+shadow (the default "bold highlight" look),
    3 = an opaque background box behind the text (the "boxed" look).

    `emphasis_words` (matched case-insensitively, punctuation-stripped) are
    user-picked keywords that get the highlight-color treatment every time
    they appear, not just when they're the currently-spoken word — e.g.
    picking "profit" colors every occurrence of "profit"/"profits" through
    the whole clip, on top of the normal per-word speaking highlight."""
    emphasis_words = {w.lower().strip(".,!?;:\"'") for w in (emphasis_words or set())}
    align = _ALIGN.get(position, 2)
    margin_v = int(video_height * 0.16) if position != "center" else int(video_height * 0.42)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{font_size},{base_color},{base_color},{outline_color},{back_color},1,0,0,0,100,100,0,0,{border_style},{outline_width},{shadow_width},{align},50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    relative_words = [
        Word(word=_clean(w.word), start=w.start - clip_start, end=w.end - clip_start, prob=w.prob)
        for w in words
    ]
    relative_words = [w for w in relative_words if w.end > 0]

    for chunk in _group_words(relative_words):
        chunk_text_parts = [w.word for w in chunk]
        for i, active in enumerate(chunk):
            rendered = []
            for j, w in enumerate(chunk):
                is_speaking_now = j == i
                is_emphasized_keyword = w.word.lower().strip(".,!?;:\"'") in emphasis_words
                if is_speaking_now or is_emphasized_keyword:
                    rendered.append(f"{{\\c{highlight_color}}}{w.word}{{\\c{base_color}}}")
                else:
                    rendered.append(w.word)
            text = " ".join(rendered)
            lines.append(
                f"Dialogue: 0,{_ts(active.start)},{_ts(active.end)},Caption,,0,0,0,,{text}"
            )

    return header + "\n".join(lines) + "\n"


def write_ass_file(ass_content: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(ass_content, encoding="utf-8")
    return output_path


def burn_subtitles(input_path: str | Path, ass_path: str | Path, output_path: str | Path) -> None:
    """Burns the .ass subtitle track into the video. Kept as its own ffmpeg
    pass (rather than trying to cram every filter into one command) so each
    pipeline stage is independently testable and debuggable."""
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        get_ffmpeg(require_ass=True), "-y", "-i", str(input_path),
        "-vf", f"ass={ass_escaped}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        str(output_path),
    ]
    logger.info("Burning captions: %s", ass_path)
    subprocess.run(cmd, capture_output=True, text=True, check=True)
