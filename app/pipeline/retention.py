"""Picks the single most-replayed window out of YouTube's own "most replayed"
retention heatmap (what yt-dlp hands back as `info["heatmap"]` — the same
data that draws the gray intensity graph under the scrubber on youtube.com),
so a clip from that moment can be surfaced to the user as "the part viewers
replay most" instead of only relying on our own local heuristics/LLM.

Only available for YouTube sources with enough view history for YouTube to
have computed the graph at all — everything here degrades to "no heatmap,
skip this" rather than erroring, since it's a bonus signal, not a
requirement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetentionWindow:
    start: float
    end: float
    avg_intensity: float  # 0-1ish, YouTube's own relative-replay scale

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_most_replayed_window(
    heatmap: list[dict] | None,
    target_duration: float,
    min_duration: float,
    max_duration: float,
) -> RetentionWindow | None:
    """Slides a window sized close to `target_duration` (clamped to
    [min_duration, max_duration]) across the heatmap and returns the
    position with the highest average "value" (replay intensity).

    `heatmap` entries are equal-width segments covering the full video, each
    shaped like {"start_time": float, "end_time": float, "value": float}.
    Returns None if there's no usable heatmap (missing, empty, or too few
    segments to form a sensible window).
    """
    if not heatmap or len(heatmap) < 2:
        return None

    segments = sorted(
        (
            (float(s["start_time"]), float(s["end_time"]), float(s.get("value") or 0.0))
            for s in heatmap
            if "start_time" in s and "end_time" in s
        ),
        key=lambda s: s[0],
    )
    if len(segments) < 2:
        return None

    total_duration = segments[-1][1]
    seg_width = (total_duration - segments[0][0]) / len(segments)
    if seg_width <= 0:
        return None

    target_duration = max(min_duration, min(max_duration, target_duration))
    window_segs = max(1, round(target_duration / seg_width))
    window_segs = min(window_segs, len(segments))

    # Expand/shrink by one segment at a time if the resulting duration would
    # fall outside the configured bounds (rare — only matters for very few
    # or very wide heatmap segments).
    def _window_duration(n: int) -> float:
        return n * seg_width

    while _window_duration(window_segs) < min_duration and window_segs < len(segments):
        window_segs += 1
    while _window_duration(window_segs) > max_duration and window_segs > 1:
        window_segs -= 1

    best_start_idx = 0
    best_sum = None
    running_sum = sum(v for _, _, v in segments[:window_segs])
    best_sum = running_sum
    for i in range(1, len(segments) - window_segs + 1):
        running_sum += segments[i + window_segs - 1][2] - segments[i - 1][2]
        if running_sum > best_sum:
            best_sum = running_sum
            best_start_idx = i

    start_time = segments[best_start_idx][0]
    end_time = segments[best_start_idx + window_segs - 1][1]
    avg_intensity = best_sum / window_segs

    return RetentionWindow(start=start_time, end=end_time, avg_intensity=avg_intensity)
