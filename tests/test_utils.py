import pytest

from app.utils import ResponseParsingError, parse_llm_json_response


def test_parse_llm_json_response_with_wrapped_text() -> None:
    raw = "Here you go:\n```json\n{\"overall_score\": 4.5, \"evaluation_type\": \"Reasoning\", \"dimensional_scores\": {}}\n```"
    result = parse_llm_json_response(raw)
    assert result["overall_score"] == 4.5


def test_parse_llm_json_response_raises_for_invalid_text() -> None:
    with pytest.raises(ResponseParsingError):
        parse_llm_json_response("No JSON here")
