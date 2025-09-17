"""Utilities for batch-evaluating interview responses from a CSV file."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import Settings, get_settings
from .llm.base import LLMProviderError
from .llm.factory import ProviderFactory, get_llm_provider
from .models import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationType,
    LLMConfig,
    ProviderType,
)
from .prompts import PromptEngine


DEFAULT_QUESTION_COLUMN = "question"
DEFAULT_ANSWER_COLUMN = "answer"
DEFAULT_IDEAL_ANSWER_COLUMN = "ideal_answer_key_points"
JSON_OUTPUT_COLUMN = "evaluation_json"
OVERALL_SCORE_COLUMN = "overall_score"


class CSVProcessingError(RuntimeError):
    """Raised when the CSV batch runner cannot process an input file."""


def _score_column_name(dimension_key: str) -> str:
    """Return a column name for a rubric dimension score."""

    sanitized = dimension_key.lower().replace(" ", "_")
    return f"score_{sanitized}"


def _load_rows(csv_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CSVProcessingError("CSV file must include a header row")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    return rows, fieldnames


def _write_rows(csv_path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_csv_file(
    csv_path: Path,
    evaluation_type: EvaluationType,
    llm_config: LLMConfig,
    *,
    question_column: str = DEFAULT_QUESTION_COLUMN,
    answer_column: str = DEFAULT_ANSWER_COLUMN,
    job_description_column: Optional[str] = None,
    resume_context_column: Optional[str] = None,
    ideal_answer_column: Optional[str] = DEFAULT_IDEAL_ANSWER_COLUMN,
    prompt_engine: Optional[PromptEngine] = None,
    provider_factory: ProviderFactory = get_llm_provider,
    settings: Optional[Settings] = None,
) -> None:
    """Evaluate each row in ``csv_path`` and enrich it with scoring details."""

    if evaluation_type == EvaluationType.CONCEPTUAL and not ideal_answer_column:
        raise CSVProcessingError(
            "Conceptual evaluations require an ideal answer column to be specified"
        )

    rows, original_columns = _load_rows(csv_path)
    if not rows:
        return

    prompt_engine = prompt_engine or PromptEngine()
    settings = settings or get_settings()

    try:
        provider = provider_factory(llm_config, settings)
    except (ValueError, LLMProviderError) as exc:  # pragma: no cover - validation handled elsewhere
        raise CSVProcessingError(str(exc)) from exc

    score_columns: List[str] = []
    for row in rows:
        question = (row.get(question_column) or "").strip()
        answer = (row.get(answer_column) or "").strip()
        if not question:
            raise CSVProcessingError(
                f"Row is missing a value in the '{question_column}' column"
            )
        if not answer:
            raise CSVProcessingError(f"Row is missing an answer for question '{question}'")

        job_description = (row.get(job_description_column) or None) if job_description_column else None
        resume_context = (row.get(resume_context_column) or None) if resume_context_column else None

        ideal_answer = None
        if evaluation_type == EvaluationType.CONCEPTUAL:
            if not ideal_answer_column:
                raise CSVProcessingError(
                    "Conceptual evaluations require an ideal answer column to be specified"
                )
            ideal_answer = (row.get(ideal_answer_column) or "").strip()
            if not ideal_answer:
                raise CSVProcessingError(
                    f"Row for question '{question}' is missing ideal answer key points"
                )

        evaluation_request = EvaluationRequest(
            evaluation_type=evaluation_type,
            question=question,
            answer=answer,
            job_description=job_description,
            resume_context=resume_context,
            ideal_answer_key_points=ideal_answer,
            llm_config=llm_config,
        )

        prompt_bundle = prompt_engine.build_prompt(evaluation_request)
        try:
            result = provider.evaluate(prompt_bundle)
        except LLMProviderError as exc:
            raise CSVProcessingError(f"LLM request failed for question '{question}': {exc}") from exc

        structured = EvaluationResponse.model_validate(result)
        if structured.raw_response is None:
            structured.raw_response = result

        dim_scores = structured.dimensional_scores
        if not dim_scores:
            raise CSVProcessingError(
                f"No dimensional scores were returned for question '{question}'"
            )

        numeric_scores = [detail.score for detail in dim_scores.values()]
        average_score = round(sum(numeric_scores) / len(numeric_scores), 2)

        json_blob = json.dumps(structured.model_dump(mode="json"), ensure_ascii=False)

        row[JSON_OUTPUT_COLUMN] = json_blob
        row[OVERALL_SCORE_COLUMN] = f"{average_score:.2f}"

        for dimension_key, detail in dim_scores.items():
            column = _score_column_name(dimension_key)
            if column not in score_columns:
                score_columns.append(column)
            row[column] = detail.score

    output_fieldnames = list(original_columns)
    for column in [JSON_OUTPUT_COLUMN, OVERALL_SCORE_COLUMN]:
        if column not in output_fieldnames:
            output_fieldnames.append(column)
    for column in score_columns:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    _write_rows(csv_path, output_fieldnames, rows)


def _preflight_check_llm_connection(llm_config: LLMConfig, settings: Settings) -> None:
    """Verify the LLM provider is reachable before processing the CSV.

    Raises:
        CSVProcessingError: when the provider cannot be reached or is misconfigured.
    """

    try:
        provider = get_llm_provider(llm_config, settings)
        provider.healthcheck()
        context = [f"provider={llm_config.provider.value}", f"model={llm_config.model_name}"]
        if llm_config.base_url is not None:
            context.append(f"base_url={llm_config.base_url}")
        print(f"LLM connectivity check succeeded ({', '.join(context)})")
    except (ValueError, LLMProviderError) as exc:
        context = [f"provider={llm_config.provider.value}", f"model={llm_config.model_name}"]
        if llm_config.base_url is not None:
            context.append(f"base_url={llm_config.base_url}")
        raise CSVProcessingError(
            f"LLM connectivity check failed ({', '.join(context)}): {exc}"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate interview answers from a CSV file.")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV file to process.")
    parser.add_argument(
        "--evaluation-type",
        required=True,
        choices=[choice.value for choice in EvaluationType],
        help="Evaluation rubric to use for all rows in the CSV.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=[choice.value for choice in ProviderType],
        help="LLM provider to use for scoring.",
    )
    parser.add_argument("--model-name", required=True, help="LLM model name to invoke.")
    parser.add_argument("--api-key", help="API key for the provider (if required).")
    parser.add_argument("--base-url", help="Base URL for LM Studio compatible providers.")
    parser.add_argument("--timeout", type=int, help="Optional timeout in seconds for each LLM call.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature for the LLM request.")
    parser.add_argument(
        "--max-output-tokens", type=int, help="Maximum number of tokens the model may generate."
    )
    parser.add_argument(
        "--question-column",
        default=DEFAULT_QUESTION_COLUMN,
        help="Name of the CSV column containing interview questions.",
    )
    parser.add_argument(
        "--answer-column",
        default=DEFAULT_ANSWER_COLUMN,
        help="Name of the CSV column containing candidate answers.",
    )
    parser.add_argument(
        "--job-description-column",
        help="Optional CSV column containing the job description for each row.",
    )
    parser.add_argument(
        "--resume-context-column",
        help="Optional CSV column containing resume context for each row.",
    )
    parser.add_argument(
        "--ideal-answer-column",
        default=DEFAULT_IDEAL_ANSWER_COLUMN,
        help="CSV column containing ideal answer key points (required for conceptual evaluations).",
    )
    return parser.parse_args()


def main() -> None:
    import sys

    try:
        args = _parse_args()
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            raise CSVProcessingError(f"CSV file not found: {csv_path}")

        evaluation_type = EvaluationType(args.evaluation_type)
        provider_type = ProviderType(args.provider)

        llm_config = LLMConfig(
            provider=provider_type,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

        # Load settings and verify LLM connectivity before processing data
        settings = get_settings()
        _preflight_check_llm_connection(llm_config, settings)

        process_csv_file(
            csv_path=csv_path,
            evaluation_type=evaluation_type,
            llm_config=llm_config,
            question_column=args.question_column,
            answer_column=args.answer_column,
            job_description_column=args.job_description_column,
            resume_context_column=args.resume_context_column,
            ideal_answer_column=args.ideal_answer_column,
            settings=settings,
        )
    except CSVProcessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
