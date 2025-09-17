"""Pydantic data models for the AI scoring service."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class EvaluationType(str, Enum):
    """Supported evaluation types."""

    REASONING = "reasoning"
    CONCEPTUAL = "conceptual"

    def display_name(self) -> str:
        """Return a human readable version of the enum value."""

        return self.value.capitalize()


class ProviderType(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    LM_STUDIO = "lm-studio"


class LLMConfig(BaseModel):
    """Configuration payload describing which LLM provider to use."""

    provider: ProviderType
    model_name: str = Field(..., min_length=1)
    api_key: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    timeout_seconds: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_provider_specific_fields(self) -> "LLMConfig":
        if self.provider == ProviderType.LM_STUDIO and self.base_url is None:
            raise ValueError("LM Studio provider requires a base_url")
        return self


class EvaluationRequest(BaseModel):
    """Incoming payload expected by the /evaluate endpoint."""

    evaluation_type: EvaluationType
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    job_description: Optional[str] = Field(default=None)
    resume_context: Optional[str] = Field(default=None)
    ideal_answer_key_points: Optional[str] = Field(default=None)
    llm_config: LLMConfig

    @model_validator(mode="after")
    def _normalise_optional_fields(self) -> "EvaluationRequest":
        """Ensure optional text fields have sensible defaults."""

        def _normalise(value: Optional[str]) -> str:
            if value is None:
                return "Not provided."
            stripped = value.strip()
            return stripped if stripped else "Not provided."

        self.job_description = _normalise(self.job_description)
        self.resume_context = _normalise(self.resume_context)
        ideal_answer = _normalise(self.ideal_answer_key_points)
        if self.evaluation_type == EvaluationType.CONCEPTUAL and ideal_answer == "Not provided.":
            raise ValueError("ideal_answer_key_points is required for conceptual evaluations")
        self.ideal_answer_key_points = ideal_answer
        return self


class ScoreDetail(BaseModel):
    """A single rubric dimension score."""

    score: float = Field(..., ge=0.0)
    analysis: str

    model_config = ConfigDict(extra="allow")


class EvaluationResponse(BaseModel):
    """Structured evaluation returned by the service."""

    evaluation_type: str
    overall_score: float = Field(..., ge=0.0)
    dimensional_scores: Dict[str, ScoreDetail]
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class HealthResponse(BaseModel):
    """Response model for the service healthcheck."""

    status: str = "ok"
    service: str = "ai-scoring"
