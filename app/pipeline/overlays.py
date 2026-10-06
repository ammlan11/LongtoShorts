"""Finds numeric/stat mentions in a clip's transcript and turns them into
short-lived on-screen graphics — a "stat card" for a single number, or a
small comparison bar chart when two comparable numbers appear close
together (e.g. "went from $10k to $50k"). Positioned in the upper third so
they never collide with the lower-third captions.

Everything here runs locally: Pillow only, a validated color palette (see
palette.py), no image-generation API, and no heavier plotting library —
the bar chart is drawn by hand with PIL primitives, which keeps the
dependency footprint small and avoids matplotlib's much heavier (and more
build-fragile) install.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ffmpeg_util import get_ffmpeg
from .palette import get_palette
from .transcribe import Word

logger = logging.getLogger("shorts_ai.overlays")

# Bundled under app/assets/fonts/ so stat-card text renders the same way on
# every OS. Previously this loaded DejaVu Sans from hardcoded Linux system
# paths (/usr/share/fonts/...) that simply don't exist on macOS/Windows --
# on those platforms every font load silently failed and PIL fell back to
# its built-in bitmap default font, which ignores the requested size and
# renders at ~10px. At a 900x240 card size that text is essentially
# invisible, so the cards appeared to be empty rounded rectangles (and, for
# comparison charts, bare colored bars) with no visible text at all.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_BUNDLED_FONT = {
    False: _ASSETS_DIR / "DejaVuSans.ttf",
    True: _ASSETS_DIR / "DejaVuSans-Bold.ttf",
}

_NUM_RE = re.compile(
    r"(?P<sign>-)?\$?(?P<num>\d[\d,]*\.?\d*)\s*(?P<suffix>%|k|m|b|million|billion|thousand|percent)?",
    re.IGNORECASE,
)
_SUFFIX_MULT = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000,
}

# Small local English number-word parser. Whisper models often *do* write
# numbers as digits ("20%", "$5,000") because that's how their training
# captions were formatted, but smaller/faster models (tiny/base) sometimes
# leave them spelled out ("twenty percent") — especially for the round
# numbers common in spoken stats. Handling both means overlays show up
# reliably regardless of which way a given clip got transcribed.
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES) | {"and"}
_UNIT_SUFFIX_WORDS = {"percent", "dollars", "dollar", "%"}


def _words_to_number(tokens: list[str]) -> float | None:
    """Converts a run of English number words (e.g. ['twenty', 'two',
    'thousand']) into a value (22000). Returns None if the tokens don't
    parse cleanly."""
    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok == "and":
            continue
        if tok in _ONES:
            current += _ONES[tok]
            seen = True
        elif tok in _TENS:
            current += _TENS[tok]
            seen = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            if scale == 100:
                current = (current or 1) * scale
            else:
                total += (current or 1) * scale
                current = 0
            seen = True
        else:
            return None
    return (total + current) if seen else None


def _clean_token(word: str) -> str:
    return word.lower().strip(".,!?;:\"'")


@dataclass
class StatMention:
    word_index: int
    time: float
    raw: str
    value: float
    display: str
    label: str


@dataclass
class OverlayItem:
    image_path: Path
    start: float
    end: float
    x: str  # ffmpeg overlay x expression
    y: str


def _parse_value(match: re.Match) -> float | None:
    num_str = match.group("num").replace(",", "")
    if not num_str or num_str == ".":
        return None
    try:
        value = float(num_str)
    except ValueError:
        return None
    suffix = (match.group("suffix") or "").lower()
    value *= _SUFFIX_MULT.get(suffix, 1)
    if match.group("sign"):
        value = -value
    return value


def _sentence_bounds(words: list[Word], idx: int) -> tuple[int, int]:
    start = idx
    while start > 0 and not words[start - 1].word.endswith((".", "?", "!")):
        start -= 1
    end = idx
    while end < len(words) - 1 and not words[end].word.endswith((".", "?", "!")):
        end += 1
    return start, end


def _make_mention(
    words: list[Word], i: int, j: int, value: float, display: str, extra_exclude: int | None = None
) -> StatMention:
    """i..j inclusive is the span of words that make up the number itself
    (used to build the display label with the number removed). `extra_exclude`
    is one more index to drop from the label (e.g. a trailing "percent")."""
    excluded = set(range(i, j + 1)) | ({extra_exclude} if extra_exclude is not None else set())
    s, e = _sentence_bounds(words, i)
    label_words = [w.word for k, w in enumerate(words[s : e + 1], start=s) if k not in excluded][:9]
    label = " ".join(label_words).strip(" .,!?")
    if len(label) > 46:
        label = label[:43] + "..."
    return StatMention(
        word_index=i,
        time=words[i].start,
        raw=words[i].word,
        value=value,
        display=display,
        label=label or "mentioned",
    )


def _extract_digit_mentions(words: list[Word]) -> list[StatMention]:
    mentions = []
    for i, w in enumerate(words):
        m = _NUM_RE.search(w.word)
        if not m or not m.group("num"):
            continue
        has_marker = "$" in w.word or "%" in w.word or (m.group("suffix") or "")
        if not has_marker and float(m.group("num").replace(",", "") or 0) < 10:
            continue
        value = _parse_value(m)
        if value is None:
            continue
        mentions.append(_make_mention(words, i, i, value, w.word.strip(".,!?")))
    return mentions


def _extract_spelled_out_mentions(words: list[Word]) -> list[StatMention]:
    mentions = []
    i = 0
    tokens = [_clean_token(w.word) for w in words]
    while i < len(words):
        if tokens[i] not in _NUMBER_WORDS:
            i += 1
            continue
        j = i
        while j + 1 < len(words) and tokens[j + 1] in _NUMBER_WORDS:
            j += 1
        value = _words_to_number(tokens[i : j + 1])
        has_unit = j + 1 < len(words) and tokens[j + 1] in _UNIT_SUFFIX_WORDS
        unit = tokens[j + 1] if has_unit else None
        if value is not None:
            if unit is not None or value >= 10:
                display_words = " ".join(w.word.strip(".,!?") for w in words[i : j + 1])
                unit_label = {"percent": "%", "dollars": "", "dollar": "", "%": "%"}.get(unit, "")
                prefix = "$" if unit in ("dollars", "dollar") else ""
                display = f"{prefix}{value:,.0f}{unit_label}" if unit_label == "%" else (
                    f"{prefix}{value:,.0f}" if prefix else f"{display_words}"
                )
                mentions.append(
                    _make_mention(words, i, j, value, display, extra_exclude=(j + 1) if has_unit else None)
                )
        i = j + 1
    return mentions


def extract_stat_mentions(words: list[Word], clip_start: float) -> list[StatMention]:
    """Scans word-by-word text for numeric mentions worth calling out — both
    digit form ("20%", "$5,000") and spelled-out form ("twenty percent",
    "five thousand dollars"), since different Whisper model sizes render
    spoken numbers differently."""
    mentions = _extract_digit_mentions(words) + _extract_spelled_out_mentions(words)
    mentions.sort(key=lambda m: m.word_index)

    # de-duplicate overlapping detections (rare, but a digit-form regex and
    # the word-parser could in principle both fire near the same index)
    deduped: list[StatMention] = []
    for m in mentions:
        if deduped and m.word_index - deduped[-1].word_index <= 1:
            continue
        deduped.append(m)

    for m in deduped:
        m.time = m.time - clip_start
    return deduped


def _group_mentions(mentions: list[StatMention], max_gap: float = 5.0) -> list[list[StatMention]]:
    groups: list[list[StatMention]] = []
    current: list[StatMention] = []
    for m in mentions:
        if current and m.time - current[-1].time > max_gap:
            groups.append(current)
            current = []
        current.append(m)
    if current:
        groups.append(current)
    return groups


def _rounded_card(size: tuple[int, int], theme: str, radius: int = 28) -> Image.Image:
    pal = get_palette(theme)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=pal["surface"], outline=pal["border"], width=2)
    return img


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Bundled font first -- guaranteed present regardless of OS. The old
    # hardcoded Linux-only system paths are kept as a fallback in case
    # someone runs from a copy of this file without the assets/ dir, and a
    # couple of common macOS/Windows system fonts are tried too, but the
    # bundled file is what actually gets used in practice.
    candidates = [
        _BUNDLED_FONT[bold],
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(str(c), size)
        except OSError:
            continue
    logger.warning("No TrueType font could be loaded (checked bundled asset + system fallbacks); "
                    "overlay text will render as PIL's tiny built-in bitmap font.")
    return ImageFont.load_default()


def render_stat_card(mention: StatMention, theme: str, out_path: Path, width: int = 900) -> Path:
    pal = get_palette(theme)
    height = 240
    card = _rounded_card((width, height), theme)
    draw = ImageDraw.Draw(card)

    number_font = _load_font(96, bold=True)
    label_font = _load_font(38)

    number_text = mention.display
    draw.text((48, 34), number_text, font=number_font, fill=pal["series_1"])
    draw.text((48, 160), mention.label, font=label_font, fill=pal["secondary_ink"])

    card.save(out_path)
    return out_path


def render_comparison_chart(mentions: list[StatMention], theme: str, out_path: Path, width: int = 900) -> Path:
    """Hand-drawn two-bar comparison chart — no plotting library involved.
    Simple by design: two values, two colors from the fixed categorical
    order (series_1, series_2), direct value labels above each bar, and the
    surrounding sentence as a title. That's the whole vocabulary this needs."""
    pal = get_palette(theme)
    a, b = mentions[0], mentions[1]
    values = [a.value, b.value]
    labels = [a.display, b.display]
    colors = [pal["series_1"], pal["series_2"]]

    height = 300
    card = _rounded_card((width, height), theme)
    draw = ImageDraw.Draw(card)

    title_font = _load_font(30)
    value_font = _load_font(40, bold=True)

    title = (mentions[0].label or "Comparison")[:48]
    draw.text((40, 28), title, font=title_font, fill=pal["secondary_ink"])

    plot_top, plot_bottom = 115, height - 36
    plot_h = plot_bottom - plot_top
    bar_w = 170
    gap = 120
    total_w = bar_w * 2 + gap
    x0 = (width - total_w) // 2

    max_val = max(abs(v) for v in values) or 1.0
    min_bar_h = 14  # keep near-zero values visible as a sliver, not invisible

    # One label per bar — its formatted value (already carries $/% from the
    # transcript), placed above the bar. No second, redundant axis label.
    for i, (value, label, color) in enumerate(zip(values, labels, colors)):
        bar_h = max(min_bar_h, int(plot_h * (abs(value) / max_val)))
        x1 = x0 + i * (bar_w + gap)
        x2 = x1 + bar_w
        y1 = plot_bottom - bar_h
        y2 = plot_bottom
        draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=color)

        lw = draw.textlength(label, font=value_font)
        draw.text((x1 + bar_w / 2 - lw / 2, y1 - 52), label, font=value_font, fill=pal["primary_ink"])

    draw.line([(40, plot_bottom), (width - 40, plot_bottom)], fill=pal["border"], width=2)

    card.save(out_path)
    return out_path


def build_overlay_items(
    words: list[Word],
    clip_start: float,
    clip_duration: float,
    work_dir: Path,
    theme: str = "dark",
    min_stats_for_card: int = 1,
    min_stats_for_chart: int = 2,
    card_seconds: float = 3.2,
    max_overlays: int = 3,
    target_width: int = 1080,
) -> list[OverlayItem]:
    work_dir.mkdir(parents=True, exist_ok=True)
    mentions = extract_stat_mentions(words, clip_start)
    if not mentions:
        return []

    groups = _group_mentions(mentions)
    items: list[OverlayItem] = []
    last_end = -999.0

    for gi, group in enumerate(groups):
        if len(items) >= max_overlays:
            break
        start_time = max(group[0].time, last_end + 0.4)
        if start_time >= clip_duration - 0.5:
            continue
        end_time = min(start_time + card_seconds, clip_duration - 0.1)
        if end_time - start_time < 1.0:
            continue

        png_path = work_dir / f"overlay_{gi}.png"
        if len(group) >= min_stats_for_chart:
            render_comparison_chart(group[:2], theme, png_path, width=int(target_width * 0.85))
        elif len(group) >= min_stats_for_card:
            render_stat_card(group[0], theme, png_path, width=int(target_width * 0.85))
        else:
            continue

        items.append(
            OverlayItem(
                image_path=png_path,
                start=start_time,
                end=end_time,
                x="(main_w-overlay_w)/2",
                y="(main_h*0.10)",  # upper safe zone, clear of lower-third captions
            )
        )
        last_end = end_time

    return items


def composite_overlays(input_video: str | Path, items: list[OverlayItem], output_path: str | Path) -> None:
    if not items:
        # nothing to do — just copy through
        subprocess.run(
            [get_ffmpeg(), "-y", "-i", str(input_video), "-c", "copy", str(output_path)],
            capture_output=True, text=True, check=True,
        )
        return

    fade = 0.25
    cmd = [get_ffmpeg(), "-y", "-i", str(input_video)]
    for item in items:
        dur = item.end - item.start
        cmd += ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(item.image_path)]

    filter_parts = []
    last_label = "0:v"
    for i, item in enumerate(items, start=1):
        dur = item.end - item.start
        fade_out_start = max(0.0, dur - fade)
        faded = f"v{i}"
        filter_parts.append(
            f"[{i}:v]format=rgba,fade=t=in:st=0:d={fade:.2f}:alpha=1,"
            f"fade=t=out:st={fade_out_start:.2f}:d={fade:.2f}:alpha=1[{faded}]"
        )
        out_label = f"ov{i}"
        filter_parts.append(
            f"[{last_label}][{faded}]overlay=x={item.x}:y={item.y}:"
            f"enable='between(t,{item.start:.3f},{item.end:.3f})'[{out_label}]"
        )
        last_label = out_label

    filter_complex = ";".join(filter_parts)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{last_label}]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        str(output_path),
    ]
    logger.info("Compositing %d overlay(s) onto %s", len(items), input_video)
    subprocess.run(cmd, capture_output=True, text=True, check=True)
