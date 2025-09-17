"""LLM provider implementation for LM Studio compatible endpoints."""
from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI

from ..config import Settings
from ..models import LLMConfig
from ..prompts import PromptBundle
from ..utils import ResponseParsingError, parse_llm_json_response
from .base import LLMProvider, LLMProviderError


class LMStudioProvider(LLMProvider):
    """Interact with an LM Studio server that mimics the OpenAI API."""

    def __init__(self, config: LLMConfig, settings: Settings) -> None:
        super().__init__(config)
        if config.base_url is None:
            raise LLMProviderError("LM Studio provider requires a base_url")
        base_url = str(config.base_url).rstrip("/")
        api_key = config.api_key or settings.openai_api_key or "lm-studio"
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._settings = settings

    def evaluate(self, prompt: PromptBundle) -> Dict[str, Any]:
        timeout = self._resolve_timeout(self._settings.default_timeout_seconds)
        client = self._client.with_options(timeout=timeout)
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self.config.temperature or 0.2,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover - upstream errors are surfaced
            raise LLMProviderError(f"LM Studio request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMProviderError("LM Studio returned an unexpected payload") from exc

        try:
            return parse_llm_json_response(content)
        except ResponseParsingError as exc:
            raise LLMProviderError(str(exc)) from exc
