# AI Interview Scoring Service

This project provides a containerised FastAPI service that scores interview answers using Large Language Models. It supports Google Gemini and LM Studio compatible providers, selects an evaluation rubric based on the question type, and returns structured JSON feedback.

## Features

- FastAPI endpoint `POST /evaluate` for scoring interview responses.
- Prompt engine that selects a reasoning or conceptual rubric and enforces JSON outputs.
- Pluggable provider architecture with implementations for Gemini and LM Studio.
- Robust parsing of model outputs with detailed error handling.
- Docker image for reproducible deployment.

### Request Payload

The `POST /evaluate` endpoint accepts a JSON body with the following fields:

- `evaluation_type`: either `reasoning` or `conceptual`.
- `question`: the interview question text.
- `answer`: the candidate's response.
- `job_description`: optional context about the role.
- `resume_context`: optional highlights from the candidate's background.
- `ideal_answer_key_points`: required when `evaluation_type` is `conceptual`; provide the canonical facts or bullet points the answer should cover.
- `llm_config`: provider configuration containing the model name and credentials.

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) or `pip`

### Installation

```bash
pip install -r requirements.txt
```

### Running Locally

1. Export the necessary API keys:

```bash
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="optional-lm-studio-key"
```

2. Start the service:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive documentation is available at `/docs`.

### Testing

```bash
pytest
```

### Batch CSV Evaluation

Use the bundled client script to score multiple interview answers stored in a CSV file. The file must include headers for at least a question and answer column, and all rows must share the same evaluation type (either reasoning or conceptual).

```bash
python -m app.csv_client \
  --evaluation-type conceptual \
  --provider lm-studio \
  --model-name llama-3 \
  --base-url http://localhost:1234/v1\
  --csv-path data/responses.csv 
```

The script processes each row sequentially, calling the configured LLM provider, and writes the returned JSON payload to an `evaluation_json` column. It also appends an `overall_score` column—calculated as the average of the dimensional scores—and creates individual `score_<dimension>` columns (five dimensions for conceptual files, six for reasoning files).


### Docker

Populate the values in `.env`, then let Docker Compose handle the build and runtime configuration:

```bash
docker compose up --build
```

The service will listen on `http://localhost:8000` and automatically receive the environment variables defined in `.env`. When you're done, stop the stack with `docker compose down`.

## Configuration

Environment variables can be provided via a `.env` file or directly in the environment. The following variables are supported:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `REQUEST_TIMEOUT` (seconds)

## Project Layout

- `app/main.py` – FastAPI application definition.
- `app/prompts.py` – Prompt templates and prompt engine.
- `app/llm/` – Provider implementations and factory helpers.
- `app/utils.py` – JSON parsing helpers.
- `tests/` – Unit tests.

## License

This project is provided as-is without any specific license.
