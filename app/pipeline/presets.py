"""Caption style and aspect ratio presets, surfaced in the UI's review step
so you can pick a look per render without hand-editing config.yaml.

Caption colors are ASS &HAABBGGRR hex (alpha, then blue/green/red — yes,
reversed from the usual #RRGGBB). The three presets below cover the common
short-form looks: a bold highlighted-word style, a plain clean style, and a
solid background box.
"""
from __future__ import annotations

CAPTION_PRESETS: dict[str, dict] = {
    "bold-highlight": {
        "label": "Bold highlight",
        "description": "White text, the current word pops in amber. The classic short-form look.",
        "font": "DejaVu Sans",
        "font_size": 64,
        "highlight_color": "&H0026D9D9",
        "base_color": "&H00FFFFFF",
        "outline_color": "&H00161616",
        "back_color": "&H64000000",
        "position": "lower_third",
        "border_style": 1,
        "outline_width": 5,
        "shadow_width": 2,
    },
    "clean-white": {
        "label": "Clean white",
        "description": "Plain bold white text, no per-word color change, thinner outline.",
        "font": "DejaVu Sans",
        "font_size": 58,
        "highlight_color": "&H00FFFFFF",
        "base_color": "&H00FFFFFF",
        "outline_color": "&H00161616",
        "back_color": "&H64000000",
        "position": "lower_third",
        "border_style": 1,
        "outline_width": 3,
        "shadow_width": 1,
    },
    "boxed": {
        "label": "Boxed",
        "description": "White text on a solid dark background box — no outline needed.",
        "font": "DejaVu Sans",
        "font_size": 60,
        "highlight_color": "&H0026D9D9",
        "base_color": "&H00FFFFFF",
        "outline_color": "&H00161616",
        "back_color": "&HD0161616",
        "position": "lower_third",
        "border_style": 3,
        "outline_width": 0,
        "shadow_width": 0,
    },
    "minimal-top": {
        "label": "Minimal top",
        "description": "Small plain white text along the top, no highlight color — stays out of the way.",
        "font": "DejaVu Sans",
        "font_size": 48,
        "highlight_color": "&H00FFFFFF",
        "base_color": "&H00FFFFFF",
        "outline_color": "&H00161616",
        "back_color": "&H64000000",
        "position": "top",
        "border_style": 1,
        "outline_width": 3,
        "shadow_width": 1,
    },
    "neon-pop": {
        "label": "Neon pop",
        "description": "Big centered text, hot-pink highlight word — high-energy look.",
        "font": "DejaVu Sans",
        "font_size": 78,
        "highlight_color": "&H00E4157D",
        "base_color": "&H00FFFFFF",
        "outline_color": "&H00161616",
        "back_color": "&H64000000",
        "position": "center",
        "border_style": 1,
        "outline_width": 6,
        "shadow_width": 2,
    },
}

DEFAULT_CAPTION_PRESET = "bold-highlight"

# Sent by the frontend instead of a real preset id when the user has set
# their own colors/size/position in the "Custom" controls rather than
# picking a canned look. get_caption_preset() falls back to the default
# preset's fields for this id, which start.render() in jobs.py then
# overrides field-by-field from the request's caption_overrides.
CUSTOM_PRESET_ID = "custom"

ASPECT_PRESETS: dict[str, dict] = {
    "9:16": {
        "label": "9:16 — Shorts / Reels / TikTok",
        "width": 1080,
        "height": 1920,
    },
    "1:1": {
        "label": "1:1 — Square (feed post)",
        "width": 1080,
        "height": 1080,
    },
    "4:5": {
        "label": "4:5 — Instagram portrait",
        "width": 1080,
        "height": 1350,
    },
}

DEFAULT_ASPECT_PRESET = "9:16"


def get_caption_preset(name: str | None) -> dict:
    return CAPTION_PRESETS.get(name or "", CAPTION_PRESETS[DEFAULT_CAPTION_PRESET])


def get_aspect_preset(name: str | None) -> dict:
    return ASPECT_PRESETS.get(name or "", ASPECT_PRESETS[DEFAULT_ASPECT_PRESET])


def hex_to_ass(hex_color: str, alpha_hex: str = "00") -> str:
    """Converts a standard "#RRGGBB" (or "RRGGBB") color, as produced by an
    HTML <input type="color">, into libass's "&HAABBGGRR" hex format — note
    the byte order is reversed (BGR, not RGB) and alpha is "00" for fully
    opaque, "FF" for fully transparent. Falls back to opaque white on
    anything that doesn't parse, rather than raising, since this only ever
    feeds a cosmetic style field."""
    h = (hex_color or "").lstrip("#").strip()
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        return "&H00FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha_hex.upper()}{b.upper()}{g.upper()}{r.upper()}"
