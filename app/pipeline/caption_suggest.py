"""Suggests a punchy title/caption + hashtags for an already-picked short,
using the clip's own transcript text.

Two paths, same shape of result:
  - Local LLM (Ollama), when available — same integration the highlight
    picker already optionally uses (see llm.py). Fully local/offline, just
    needs Ollama installed and running with a model pulled.
  - A zero-dependency heuristic fallback, so this still produces *something*
    useful with nothing installed beyond this app itself.

Nothing here calls out to a paid/cloud API — consistent with the rest of
this project staying fully local.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from .highlights import _STOPWORDS  # reuse the same "not a topic word" list
from .llm import extract_json, generate, is_available

logger = logging.getLogger("shorts_ai.caption_suggest")

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def _heuristic_hashtags(text: str, limit: int = 5) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    counts = Counter(t for t in tokens if len(t) > 3 and t not in _STOPWORDS)
    ranked = [w for w, _ in counts.most_common(limit * 2)]
    # Preserve first-seen order among the most common words so hashtags read
    # roughly in the order the clip actually mentions them.
    seen_order = [t for t in tokens if t in ranked]
    ordered = list(dict.fromkeys(seen_order))[:limit]
    return [f"#{w}" for w in (ordered or ranked[:limit])]


def _heuristic_suggestions(text: str) -> list[dict]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    first = sentences[0] if sentences else text[:70]
    hashtags = _heuristic_hashtags(text)

    def _cap(s: str, limit: int = 70) -> str:
        s = s.strip(" .,!?")
        return (s[: limit - 1] + "…") if len(s) > limit else s

    suggestions = [{"caption": _cap(first), "hashtags": hashtags}]
    if len(sentences) > 1:
        suggestions.append({"caption": _cap(sentences[1]), "hashtags": hashtags})
    # A generic "question hook" fallback variant so there are always a
    # couple of distinct-feeling options even for very short transcripts.
    suggestions.append({"caption": _cap(f"Watch this: {first}"), "hashtags": hashtags})
    return suggestions[:3]


def suggest_captions(
    text: str,
    use_llm: bool = True,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.1",
) -> list[dict]:
    """Returns up to 3 {"caption": str, "hashtags": [str, ...]} suggestions
    for this clip's transcript text. Tries the local LLM first (if enabled
    and reachable), falling back to a heuristic extraction on any failure —
    callers never need to handle "no suggestions" as a special case."""
    text = (text or "").strip()
    if not text:
        return []

    if use_llm and is_available(ollama_url):
        prompt = f"""You are writing the on-screen title/caption and hashtags for a YouTube Short.
Here is the Short's spoken transcript:
\"\"\"{text}\"\"\"

Suggest 3 different punchy, scroll-stopping captions (each under 70 characters, no quotes \
around them) a creator could use for this Short, plus 3-5 relevant hashtags (no spaces, \
lowercase, include the # symbol) shared across them.

Respond with ONLY a JSON array, no prose:
[{{"caption": "<caption text>", "hashtags": ["#tag1", "#tag2", "#tag3"]}}, ...]
"""
        raw = generate(prompt, model=ollama_model, base_url=ollama_url)
        data = extract_json(raw) if raw else None
        if isinstance(data, list) and data:
            cleaned = []
            for item in data:
                if not isinstance(item, dict) or not item.get("caption"):
                    continue
                hashtags = item.get("hashtags") or []
                hashtags = [h if h.startswith("#") else f"#{h}" for h in hashtags if h]
                cleaned.append({"caption": str(item["caption"])[:100], "hashtags": hashtags[:6]})
            if cleaned:
                return cleaned[:3]
        logger.info("LLM caption suggestion unavailable/unparseable, falling back to heuristic")

    return _heuristic_suggestions(text)
