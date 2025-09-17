import pytest
from pydantic import ValidationError

from app.models import EvaluationRequest, EvaluationType, LLMConfig, ProviderType
from app.prompts import PromptEngine


def _sample_request(evaluation_type: EvaluationType = EvaluationType.REASONING) -> EvaluationRequest:
    ideal_answer = None
    if evaluation_type == EvaluationType.CONCEPTUAL:
        ideal_answer = "Key metrics and guardrails for onboarding success."
    return EvaluationRequest(
        evaluation_type=evaluation_type,
        question="How would you approach improving onboarding?",
        answer="I would map the funnel, test hypotheses, and prioritise quick wins.",
        job_description="Drive product growth",
        resume_context="Growth lead with experimentation background",
        ideal_answer_key_points=ideal_answer,
        llm_config=LLMConfig(
            provider=ProviderType.LM_STUDIO,
            model_name="test-model",
            base_url="http://localhost:8000/v1",
        ),
    )


def test_build_prompt_includes_context() -> None:
    engine = PromptEngine()
    prompt = engine.build_prompt(_sample_request())
    assert "Growth lead with experimentation background" in prompt.user
    assert '"evaluation_type": "Reasoning"' in prompt.user


def test_build_prompt_for_conceptual_type() -> None:
    engine = PromptEngine()
    prompt = engine.build_prompt(_sample_request(EvaluationType.CONCEPTUAL))
    assert "IDEAL_ANSWER_KEY_POINTS" in prompt.user
    assert "Key metrics and guardrails for onboarding success." in prompt.user


def test_conceptual_request_requires_ideal_answer() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest(
            evaluation_type=EvaluationType.CONCEPTUAL,
            question="Explain CAP theorem.",
            answer="It balances trade-offs.",
            job_description="Distributed systems role",
            resume_context="Worked on databases",
            llm_config=LLMConfig(
                provider=ProviderType.LM_STUDIO,
                model_name="test-model",
                base_url="http://localhost:8000/v1",
            ),
        )
