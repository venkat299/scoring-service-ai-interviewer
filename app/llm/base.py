"""Abstract base classes and common utilities for LLM providers."""
from __future__ import annotations

import abc
from typing import Any, Dict

from ..models import LLMConfig
from ..prompts import PromptBundle


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider fails to return a valid response."""


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM provider implementations."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def evaluate(self, prompt: PromptBundle) -> Dict[str, Any]:
        """Execute the prompt and return the parsed JSON structure."""

    def _resolve_timeout(self, default_timeout: int) -> int:
        """Return the timeout to use for outbound requests."""

        return self.config.timeout_seconds or default_timeout
