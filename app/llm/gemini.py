"""LLM provider implementation for the Google Gemini API."""
from __future__ import annotations

from typing import Any, Dict, Optional

import google.generativeai as genai

from ..config import Settings
from ..models import LLMConfig
from ..prompts import PromptBundle
from ..utils import ResponseParsingError, parse_llm_json_response
from .base import LLMProvider, LLMProviderError


class GeminiProvider(LLMProvider):
    """Send evaluation prompts to the Gemini API."""

    def __init__(self, config: LLMConfig, settings: Settings) -> None:
        super().__init__(config)
        api_key = config.api_key or settings.gemini_api_key
        if not api_key:
            raise LLMProviderError("Gemini API key not provided")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(config.model_name)
        self._settings = settings

    def _build_generation_config(self) -> Optional[Dict[str, Any]]:
        temperature = self.config.temperature or 0.2
        config: Dict[str, Any] = {"temperature": temperature}
        if self.config.max_output_tokens is not None:
            config["max_output_tokens"] = self.config.max_output_tokens
        return config

    def evaluate(self, prompt: PromptBundle) -> Dict[str, Any]:
        timeout = self._resolve_timeout(self._settings.default_timeout_seconds)
        generation_config = self._build_generation_config()
        try:
            response = self._model.generate_content(
                prompt.as_text(),
                generation_config=generation_config,
                request_options={"timeout": timeout},
            )
        except Exception as exc:  # pragma: no cover - upstream errors are surfaced
            raise LLMProviderError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMProviderError("Gemini returned an empty response")

        try:
            return parse_llm_json_response(text)
        except ResponseParsingError as exc:
            raise LLMProviderError(str(exc)) from exc

    def healthcheck(self) -> None:
        """Perform a lightweight connectivity check against the Gemini API."""
        try:
            # count_tokens is lightweight and validates credentials + model access
            self._model.count_tokens("ping")
        except Exception as exc:  # pragma: no cover - upstream errors are surfaced
            raise LLMProviderError(f"Gemini connectivity check failed: {exc}") from exc
