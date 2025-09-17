"""Factory helpers to construct LLM provider instances."""
from __future__ import annotations

from typing import Callable

from ..config import Settings
from ..models import LLMConfig, ProviderType
from .base import LLMProvider
from .gemini import GeminiProvider
from .lm_studio import LMStudioProvider


ProviderFactory = Callable[[LLMConfig, Settings], LLMProvider]


def get_llm_provider(config: LLMConfig, settings: Settings) -> LLMProvider:
    """Instantiate the correct provider for the supplied configuration."""

    if config.provider == ProviderType.GEMINI:
        return GeminiProvider(config, settings)
    if config.provider == ProviderType.LM_STUDIO:
        return LMStudioProvider(config, settings)
    raise ValueError(f"Unsupported provider: {config.provider}")
