"""Thin client for a local LLM via Ollama (https://ollama.com).

This is entirely optional. Every caller in this project treats the LLM as an
optional *refinement* step — if Ollama isn't installed or isn't running, the
pipeline still works end-to-end using pure heuristics. That's the point of
"open-source/local models": no API keys, no per-request cost, and the tool
degrades gracefully to zero external dependencies if you never install a
local model.

To enable it:
    1. Install Ollama: https://ollama.com/download
    2. `ollama pull llama3.1` (or any instruction-following model you like)
    3. Make sure `ollama serve` is running (it usually auto-starts)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger("shorts_ai.llm")

_availability_cache: dict[str, bool] = {}


def is_available(base_url: str, timeout: float = 1.5) -> bool:
    if base_url in _availability_cache:
        return _availability_cache[base_url]
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        ok = r.status_code == 200
    except requests.RequestException:
        ok = False
    _availability_cache[base_url] = ok
    if not ok:
        logger.info("Ollama not reachable at %s — continuing with heuristics only", base_url)
    return ok


def generate(
    prompt: str,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    timeout: float = 60.0,
    json_mode: bool = False,
) -> Optional[str]:
    """Returns the model's raw text response, or None if the LLM is unavailable
    or errors out. Callers must handle None."""
    if not is_available(base_url):
        return None
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response")
    except requests.RequestException as e:
        logger.warning("Ollama generate() failed: %s", e)
        return None


def extract_json(text: str) -> Optional[dict | list]:
    """LLMs love to wrap JSON in prose or code fences. Pull out the first
    top-level JSON object/array and parse it."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    match = re.search(r"[\[{].*[\]}]", candidate, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
