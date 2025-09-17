"""Utility helpers for the scoring service."""
from __future__ import annotations

import json
import re
from typing import Any, Dict


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class ResponseParsingError(RuntimeError):
    """Raised when an LLM response cannot be coerced into JSON."""


def extract_json_block(raw_text: str) -> str:
    """Attempt to locate the first JSON object within ``raw_text``."""

    match = JSON_BLOCK_PATTERN.search(raw_text)
    if not match:
        raise ResponseParsingError("No JSON object found in LLM response")
    return match.group(0)


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON object from the ``raw_text`` returned by a model."""

    cleaned = raw_text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_fragment = extract_json_block(cleaned)
        try:
            return json.loads(json_fragment)
        except json.JSONDecodeError as exc:
            raise ResponseParsingError("Unable to parse LLM JSON response") from exc
