import pytest

from app.config import Settings
from app.llm.base import LLMProviderError
from app.llm.gemini import GeminiProvider
from app.models import LLMConfig, ProviderType
from app.prompts import PromptBundle


def test_gemini_provider_requires_api_key() -> None:
    config = LLMConfig(provider=ProviderType.GEMINI, model_name="gemini-test")
    settings = Settings(default_timeout_seconds=30)
    with pytest.raises(LLMProviderError):
        GeminiProvider(config, settings)


def test_gemini_provider_evaluate_parses_json(monkeypatch) -> None:
    captured = {}

    def fake_configure(api_key: str) -> None:
        captured["api_key"] = api_key

    class DummyResponse:
        text = '{"evaluation_type":"Reasoning","overall_score":4.0,"dimensional_scores":{}}'

    class DummyModel:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

        def generate_content(self, prompt_text: str, generation_config=None, request_options=None):
            captured["prompt_text"] = prompt_text
            captured["generation_config"] = generation_config
            captured["request_options"] = request_options
            return DummyResponse()

    monkeypatch.setattr("app.llm.gemini.genai.configure", fake_configure)
    monkeypatch.setattr("app.llm.gemini.genai.GenerativeModel", DummyModel)

    config = LLMConfig(
        provider=ProviderType.GEMINI,
        model_name="gemini-pro",
        api_key="api-token",
        max_output_tokens=256,
    )
    settings = Settings(default_timeout_seconds=45)

    provider = GeminiProvider(config, settings)
    result = provider.evaluate(PromptBundle(system="sys", user="user"))

    assert result["overall_score"] == 4.0
    assert captured["api_key"] == "api-token"
    assert captured["model_name"] == "gemini-pro"
    assert captured["request_options"]["timeout"] == 45
    assert captured["generation_config"]["max_output_tokens"] == 256
