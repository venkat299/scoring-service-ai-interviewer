from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMProviderError
from app.main import app, get_provider_factory


class DummyProvider:
    def __init__(self, response: Dict[str, Any], *, should_fail: bool = False) -> None:
        self._response = response
        self._should_fail = should_fail
        self.last_prompt = None

    def evaluate(self, prompt) -> Dict[str, Any]:
        self.last_prompt = prompt
        if self._should_fail:
            raise LLMProviderError("provider failure")
        return self._response


def _build_payload() -> Dict[str, Any]:
    return {
        "evaluation_type": "reasoning",
        "question": "How would you reduce churn?",
        "answer": "I'd segment users and experiment with messaging.",
        "job_description": "Lead retention strategy",
        "resume_context": "Scaled retention at SaaS startup",
        "llm_config": {
            "provider": "lm-studio",
            "model_name": "local-model",
            "base_url": "http://localhost:1234/v1",
        },
    }


def test_evaluate_endpoint_success(monkeypatch) -> None:
    response_payload = {
        "evaluation_type": "Reasoning",
        "overall_score": 4.2,
        "dimensional_scores": {
            "problem_comprehension": {"score": 4.5, "analysis": "Understood"},
            "structured_thinking": {"score": 4.0, "analysis": "Clear steps"},
        },
        "strengths": ["Structured"],
        "risks": ["Needs more data"],
        "recommendation": "Advance",
    }
    provider = DummyProvider(response_payload)

    def factory(config, settings):
        return provider

    monkeypatch.setitem(app.dependency_overrides, get_provider_factory, lambda: factory)

    client = TestClient(app)
    resp = client.post("/evaluate", json=_build_payload())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_score"] == pytest.approx(4.2)
    assert "raw_response" in data
    assert provider.last_prompt is not None
    assert "How would you reduce churn" in provider.last_prompt.user


def test_evaluate_endpoint_handles_provider_errors(monkeypatch) -> None:
    provider = DummyProvider({}, should_fail=True)

    def factory(config, settings):
        return provider

    monkeypatch.setitem(app.dependency_overrides, get_provider_factory, lambda: factory)

    client = TestClient(app)
    resp = client.post("/evaluate", json=_build_payload())

    app.dependency_overrides.clear()

    assert resp.status_code == 502
    assert resp.json()["detail"] == "provider failure"
