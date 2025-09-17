import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.csv_client import (
    CSVProcessingError,
    JSON_OUTPUT_COLUMN,
    OVERALL_SCORE_COLUMN,
    process_csv_file,
)
from app.models import EvaluationType, LLMConfig, ProviderType


class DummyProvider:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = responses
        self._calls = 0

    def evaluate(self, prompt: Any) -> Dict[str, Any]:  # pragma: no cover - prompt shape not asserted
        try:
            response = self._responses[self._calls]
        except IndexError as exc:  # pragma: no cover - guard for misconfigured tests
            raise AssertionError("Provider called more times than expected") from exc
        self._calls += 1
        return response


def _provider_factory(responses: List[Dict[str, Any]]):
    def _factory(config: LLMConfig, settings: Any) -> DummyProvider:  # pragma: no cover - interface stub
        return DummyProvider(responses)

    return _factory


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_process_csv_writes_scores_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "batch.csv"
    fieldnames = ["question", "answer", "ideal_answer_key_points"]
    rows = [
        {
            "question": "Explain eventual consistency.",
            "answer": "It means writes propagate asynchronously.",
            "ideal_answer_key_points": "Covers replication lag, client implications, mitigation strategies.",
        },
        {
            "question": "What is a quiescent state?",
            "answer": "When the system has processed all events.",
            "ideal_answer_key_points": "All activities settled, useful in distributed testing.",
        },
    ]
    _write_csv(csv_path, fieldnames, rows)

    responses: List[Dict[str, Any]] = [
        {
            "evaluation_type": "Conceptual",
            "overall_score": 4.1,
            "dimensional_scores": {
                "factual_accuracy": {"analysis": "Accurate description", "score": 4},
                "depth_of_knowledge": {"analysis": "Good context", "score": 5},
                "clarity_of_explanation": {"analysis": "Clear", "score": 4},
                "practical_application": {"analysis": "Links to SLAs", "score": 4},
                "handling_of_nuance": {"analysis": "Mentions trade-offs", "score": 3},
            },
            "strengths": ["Understands replication"],
            "risks": [],
            "recommendation": "Proceed",
        },
        {
            "evaluation_type": "Conceptual",
            "overall_score": 3.0,
            "dimensional_scores": {
                "factual_accuracy": {"analysis": "Basic", "score": 3},
                "depth_of_knowledge": {"analysis": "Surface-level", "score": 3},
                "clarity_of_explanation": {"analysis": "Concise", "score": 3},
                "practical_application": {"analysis": "Minimal", "score": 3},
                "handling_of_nuance": {"analysis": "No nuance", "score": 3},
            },
            "strengths": [],
            "risks": ["Limited depth"],
            "recommendation": "Consider follow-up",
        },
    ]

    llm_config = LLMConfig(
        provider=ProviderType.LM_STUDIO,
        model_name="dummy-model",
        base_url="http://localhost:1234/v1",
    )

    process_csv_file(
        csv_path=csv_path,
        evaluation_type=EvaluationType.CONCEPTUAL,
        llm_config=llm_config,
        provider_factory=_provider_factory(responses),
    )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        output_rows = list(reader)

    assert reader.fieldnames is not None
    assert JSON_OUTPUT_COLUMN in reader.fieldnames
    assert OVERALL_SCORE_COLUMN in reader.fieldnames
    assert "score_factual_accuracy" in reader.fieldnames

    first = output_rows[0]
    payload = json.loads(first[JSON_OUTPUT_COLUMN])
    assert payload["evaluation_type"] == "Conceptual"
    assert first[OVERALL_SCORE_COLUMN] == "4.00"
    assert float(first["score_depth_of_knowledge"]) == 5.0

    second = output_rows[1]
    assert second[OVERALL_SCORE_COLUMN] == "3.00"
    assert float(second["score_practical_application"]) == 3.0


def test_process_csv_requires_ideal_answer_for_conceptual(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    fieldnames = ["question", "answer"]
    rows = [
        {
            "question": "Define ACID.",
            "answer": "Atomicity, consistency, isolation, durability.",
        }
    ]
    _write_csv(csv_path, fieldnames, rows)

    llm_config = LLMConfig(
        provider=ProviderType.LM_STUDIO,
        model_name="dummy-model",
        base_url="http://localhost:1234/v1",
    )

    with pytest.raises(CSVProcessingError):
        process_csv_file(
            csv_path=csv_path,
            evaluation_type=EvaluationType.CONCEPTUAL,
            llm_config=llm_config,
            provider_factory=_provider_factory([]),
        )
