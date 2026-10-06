"""Colors for generated overlay graphics, taken from a pre-validated
accessible palette (colorblind-safe categorical order, checked contrast
ratios). See the project's dataviz reference for how these were derived —
the short version: don't invent new hex codes here, extend this table
following the same method instead.
"""

DARK = {
    "surface": (26, 26, 25, 218),       # #1a1a19, card background (alpha ~85%)
    "primary_ink": (255, 255, 255, 255),  # #ffffff
    "secondary_ink": (195, 194, 183, 255),  # #c3c2b7
    "muted": (137, 135, 129, 255),      # #898781
    "border": (255, 255, 255, 26),      # 10% white
    "series_1": (57, 135, 229, 255),    # blue   #3987e5
    "series_2": (217, 89, 38, 255),     # orange #d95926
    "good": (12, 163, 12, 255),         # #0ca30c
    "critical": (230, 103, 103, 255),   # #e66767
}

LIGHT = {
    "surface": (252, 252, 251, 235),    # #fcfcfb
    "primary_ink": (11, 11, 11, 255),   # #0b0b0b
    "secondary_ink": (82, 81, 78, 255), # #52514e
    "muted": (137, 135, 129, 255),      # #898781
    "border": (11, 11, 11, 26),
    "series_1": (42, 120, 214, 255),    # #2a78d6
    "series_2": (235, 104, 52, 255),    # #eb6834
    "good": (0, 99, 0, 255),            # #006300
    "critical": (211, 59, 59, 255),     # #d03b3b
}


def get_palette(theme: str = "dark") -> dict:
    return DARK if theme == "dark" else LIGHT
