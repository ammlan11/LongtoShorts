"""Finds the segments of a long video most likely to work as standalone
Shorts, using a transparent heuristic scorer (always runs) optionally
refined by a local LLM (only if Ollama is reachable — see llm.py).

Design goal: never a black box. Every selected clip carries a `reasons`
list explaining *why* it was picked, so you can tune the weights below
instead of guessing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .llm import extract_json, generate, is_available
from .transcribe import Transcript, Word

logger = logging.getLogger("shorts_ai.highlights")

_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")
_HOOK_STARTERS = (
    "how", "why", "what", "the secret", "nobody", "never", "always", "stop",
    "here's", "this is", "did you know", "the truth", "i used to", "biggest",
    "the one thing", "you're", "your", "if you",
)
_ENGAGEMENT_WORDS = (
    "secret", "mistake", "never", "always", "proven", "results", "free",
    "biggest", "worst", "best", "truth", "nobody tells you", "warning",
    "actually", "surprising", "shocking", "important", "crucial", "avoid",
    "guarantee", "risk", "profit", "loss", "win", "fail",
)
_SENTENCE_END = (".", "?", "!")

# Phrases that strongly signal "this is a preview/recap of the whole video",
# almost always sitting in the first minute or so. Picking one of these as a
# standalone Short is a bad time: it's often disjointed on its own (it
# name-drops several unrelated topics covered later) and can spoil the
# video's payoff for anyone who then watches the full thing.
_RECAP_PHRASES = (
    "in this video", "in today's video", "in today's episode",
    "by the end of this video", "by the end of this episode",
    "here's what we'll cover", "here's what you'll learn",
    "here's everything you need to know", "here's everything you'll need",
    "quick recap", "to recap", "let's recap", "coming up in this video",
    "what you're about to see", "here's a quick breakdown",
    "today i'm going to show you", "today we're going to cover",
    "in the next few minutes", "stick around", "by the time this video is over",
)
# Short, high-frequency words don't count as "topic" words when measuring how
# much an early segment's vocabulary echoes later in the video -- otherwise
# almost everything would look like a recap.
_STOPWORDS = frozenset("""
    the a an and or but so if is are was were be been being to of in on at
    for with as by from this that these those it its it's i you he she we
    they them his her our your their what which who whom not no do does did
    have has had will would can could should just about into over after
    before than then there here out up down off very really like get got
    going go went know think thing things want make made one two
""".split())


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    words: list[Word]
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    title: str = ""
    # True when this candidate is (or overlaps) the single window YouTube's
    # own "most replayed" retention graph points to for this video — see
    # pipeline/retention.py. Purely informational; doesn't change scoring.
    most_replayed: bool = False
    # True when this candidate looks like a recap/preview of the whole video
    # (see _recap_signal below) -- still scoreable and selectable, but
    # deprioritized and flagged so it doesn't get rendered without the user
    # noticing it's likely to spoil/duplicate the rest of the video.
    is_recap: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def _is_sentence_start(idx: int, words: list[Word]) -> bool:
    if idx == 0:
        return True
    prev = words[idx - 1].word
    return prev.endswith(_SENTENCE_END)


def _is_sentence_end(word: Word) -> bool:
    return word.word.endswith(_SENTENCE_END)


def _build_candidates(
    words: list[Word], min_dur: float, max_dur: float, step_sec: float = 4.0
) -> list[Candidate]:
    """Slides a window over the flat word list, snapping edges to sentence
    boundaries where possible so clips don't start/end mid-sentence."""
    candidates: list[Candidate] = []
    if not words:
        return candidates

    total_duration = words[-1].end
    n = len(words)
    t = 0.0
    # Tracks "first word index with start >= t" incrementally across the
    # whole pass instead of re-scanning from word 0 on every iteration. With
    # ~900 windows over a 60-minute video, re-scanning from the start each
    # time turns this into an O(words * windows) ~ O(video_length^2) pass --
    # fine at 17 minutes, but a real (and needless) slowdown risk as videos
    # get longer. Since both t and word start-times only increase, this
    # pointer never needs to move backward.
    scan_from = 0
    while t < total_duration:
        while scan_from < n and words[scan_from].start < t:
            scan_from += 1
        start_idx = scan_from if scan_from < n else None
        if start_idx is None:
            break
        # snap forward to the nearest sentence start within 3s
        for i in range(start_idx, min(start_idx + 15, len(words))):
            if _is_sentence_start(i, words):
                start_idx = i
                break

        target_end = words[start_idx].start + max_dur
        end_idx = start_idx
        for i in range(start_idx, len(words)):
            if words[i].end - words[start_idx].start > max_dur:
                break
            end_idx = i

        # snap backward to the nearest sentence end, as long as it still meets min_dur
        best_end_idx = end_idx
        for i in range(end_idx, start_idx - 1, -1):
            if _is_sentence_end(words[i]) and words[i].end - words[start_idx].start >= min_dur:
                best_end_idx = i
                break

        seg_words = words[start_idx : best_end_idx + 1]
        if seg_words and seg_words[-1].end - seg_words[0].start >= min_dur:
            text = " ".join(w.word for w in seg_words)
            candidates.append(
                Candidate(start=seg_words[0].start, end=seg_words[-1].end, text=text, words=seg_words)
            )
        t += step_sec

    return candidates


def _heuristic_score(c: Candidate) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    text_lower = c.text.lower()
    dur = max(c.duration, 0.1)

    # 1. Hook opener
    first_words = " ".join(w.word.lower() for w in c.words[:6])
    if any(first_words.startswith(h) or h in first_words for h in _HOOK_STARTERS):
        score += 2.0
        reasons.append("opens with a hook (question / direct address / bold claim)")
    if "?" in " ".join(w.word for w in c.words[:8]):
        score += 1.0
        reasons.append("opens with a question")

    # 2. Concrete numbers / stats — great for both retention and for the
    #    graphic-overlay stage downstream.
    numbers = _NUMBER_RE.findall(c.text)
    if numbers:
        score += min(len(numbers), 4) * 0.8
        reasons.append(f"contains {len(numbers)} concrete number(s)/stat(s)")

    # 3. Engagement / emotional-trigger words
    hits = [w for w in _ENGAGEMENT_WORDS if w in text_lower]
    if hits:
        score += min(len(hits), 5) * 0.5
        reasons.append(f"uses attention words: {', '.join(hits[:3])}")

    # 4. Speech pace — too slow (lots of dead air) or absurdly fast (ASR
    #    glitch) both hurt.
    wps = len(c.words) / dur
    if 2.0 <= wps <= 3.6:
        score += 1.5
        reasons.append("energetic, well-paced delivery")
    elif wps < 1.2:
        score -= 1.5

    # 5. Ideal short length sweet spot (~30-45s converts best on Shorts)
    if 25 <= c.duration <= 45:
        score += 1.0

    # 6. Self-contained: starts and ends on sentence boundaries
    if c.words and _is_sentence_end(c.words[-1]):
        score += 0.5
        reasons.append("ends on a clean sentence boundary")

    return score, reasons


def _content_words(words: list[Word]) -> set[str]:
    out = set()
    for w in words:
        token = w.word.lower().strip(".,!?;:\"'()")
        if len(token) > 3 and token not in _STOPWORDS:
            out.add(token)
    return out


def _recap_signal(c: Candidate, all_words: list[Word], video_duration: float) -> tuple[bool, str | None]:
    """Flags a candidate as a likely recap/preview of the whole video.

    Two independent signals, either one is enough:
      1. A direct phrase match ("in this video", "quick recap", ...).
      2. "Topic echo": a chunk of this candidate's distinctive vocabulary
         also shows up much later in the video -- the classic fingerprint of
         a cold-open that previews several unrelated moments from
         throughout the episode. A real self-contained highlight usually
         doesn't re-use its specific nouns/numbers that far downstream.

    Only ever checked near the start of the video (recap intros don't show
    up at minute 40), so this can't accidentally flag a legitimately
    self-contained clip that just happens to share common words with
    something later.
    """
    recap_window = min(90.0, video_duration * 0.2)
    if c.start > recap_window or video_duration <= 0:
        return False, None

    text_lower = c.text.lower()
    phrase_hit = next((p for p in _RECAP_PHRASES if p in text_lower), None)
    if phrase_hit:
        return True, f"opens with a preview/recap phrase (\"{phrase_hit}\")"

    my_words = _content_words(c.words)
    if len(my_words) < 4:
        return False, None
    # "later" = everything starting well after this candidate ends, so a
    # clip's own immediate continuation doesn't count against it.
    later_words = _content_words([w for w in all_words if w.start > c.end + 20.0])
    if not later_words:
        return False, None
    overlap = len(my_words & later_words) / len(my_words)
    if overlap >= 0.45:
        return True, "echoes a wide spread of topics that reappear much later in the video (looks like a preview/recap)"
    return False, None


def _overlap_ratio(a: Candidate, b: Candidate) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shortest = min(a.duration, b.duration)
    return overlap / shortest if shortest > 0 else 0.0


def _select_top_n(candidates: list[Candidate], n: int, overlap_thresh: float = 0.35) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: list[Candidate] = []
    for c in ranked:
        if len(selected) >= n:
            break
        if all(_overlap_ratio(c, s) < overlap_thresh for s in selected):
            selected.append(c)
    return sorted(selected, key=lambda c: c.start)


def _llm_refine(
    candidates: list[Candidate], clip_count: int, ollama_url: str, ollama_model: str
) -> list[Candidate] | None:
    """Asks the local LLM to pick the best `clip_count` candidates and give
    each a short punchy title + one-line reason. Returns None on any failure
    so the caller falls back to pure heuristics."""
    if not is_available(ollama_url):
        return None

    numbered = "\n".join(
        f"{i}. [{c.start:.0f}s-{c.end:.0f}s, heuristic_score={c.score:.1f}] {c.text}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""You are selecting clips from a long-form video transcript to repurpose as YouTube Shorts.
Below are {len(candidates)} candidate clips with their start/end time and text.
Pick the {clip_count} BEST candidates for standalone short-form video (must hook \
in the first sentence, be understandable with zero context, and have a clear payoff).

Candidates:
{numbered}

Respond with ONLY a JSON array, no prose, of exactly {clip_count} objects:
[{{"index": <candidate index number>, "title": "<8 word max punchy title>", "reason": "<one short sentence why this works as a short>"}}]
"""
    raw = generate(prompt, model=ollama_model, base_url=ollama_url, json_mode=False)
    data = extract_json(raw) if raw else None
    if not isinstance(data, list):
        logger.warning("LLM refine returned unparseable output, falling back to heuristics")
        return None

    picked: list[Candidate] = []
    for item in data:
        try:
            idx = int(item["index"])
            c = candidates[idx]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        c.title = item.get("title", c.title)
        if item.get("reason"):
            c.reasons.insert(0, f"LLM: {item['reason']}")
        picked.append(c)

    return picked[:clip_count] if picked else None


def find_highlights(
    transcript: Transcript,
    clip_count: int = 3,
    min_duration_sec: float = 20,
    max_duration_sec: float = 59,
    use_llm: bool = True,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.1",
) -> list[Candidate]:
    words = transcript.flat_words()
    candidates = _build_candidates(words, min_duration_sec, max_duration_sec)
    if not candidates:
        return []

    video_duration = words[-1].end if words else 0.0
    for c in candidates:
        c.score, c.reasons = _heuristic_score(c)
        is_recap, why = _recap_signal(c, words, video_duration)
        if is_recap:
            c.is_recap = True
            c.score -= 3.0
            c.reasons.insert(0, f"⚠️ Looks like a recap/preview of the whole video -- {why}")

    # Give the LLM a manageable shortlist: top 3x what we ultimately need.
    shortlist = sorted(candidates, key=lambda c: c.score, reverse=True)[: clip_count * 4]
    shortlist_dedup = _select_top_n(shortlist, min(len(shortlist), clip_count * 3), overlap_thresh=0.2)

    final: list[Candidate] | None = None
    if use_llm:
        final = _llm_refine(shortlist_dedup, clip_count, ollama_url, ollama_model)

    if not final:
        final = _select_top_n(shortlist_dedup, clip_count)
        for c in final:
            if not c.title:
                c.title = (c.text[:50] + "...") if len(c.text) > 50 else c.text

    return final
