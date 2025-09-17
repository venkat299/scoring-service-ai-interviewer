"""FastAPI entrypoint for the AI interview scoring service."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from .config import Settings, get_settings
from .llm.base import LLMProviderError
from .llm.factory import ProviderFactory, get_llm_provider
from .models import EvaluationRequest, EvaluationResponse, HealthResponse
from .prompts import PromptEngine

app = FastAPI(title="AI Interview Scoring Service", version="1.0.0")

_prompt_engine = PromptEngine()


def get_prompt_engine() -> PromptEngine:
    """FastAPI dependency returning the shared prompt engine."""

    return _prompt_engine


def get_provider_factory() -> ProviderFactory:
    """Return the callable responsible for constructing providers."""

    return get_llm_provider


@app.get("/health", response_model=HealthResponse, tags=["health"])
def healthcheck() -> HealthResponse:
    """Simple endpoint that confirms the service is running."""

    return HealthResponse()


@app.post("/evaluate", response_model=EvaluationResponse, tags=["evaluation"])
async def evaluate(
    payload: EvaluationRequest,
    settings: Settings = Depends(get_settings),
    prompt_engine: PromptEngine = Depends(get_prompt_engine),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> EvaluationResponse:
    """Evaluate an interview response using the configured LLM provider."""

    prompt_bundle = prompt_engine.build_prompt(payload)
    try:
        provider = provider_factory(payload.llm_config, settings)
    except (ValueError, LLMProviderError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await run_in_threadpool(provider.evaluate, prompt_bundle)
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        structured = EvaluationResponse.model_validate(result)
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail="Invalid response structure from provider") from exc
    if structured.raw_response is None:
        structured.raw_response = result
    return structured
