"""Prompt templates for different evaluation types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .models import EvaluationRequest, EvaluationType


SYSTEM_PROMPT = (
    "You are an expert technical interviewer tasked with evaluating candidate answers. "
    "Follow the rubric closely and return ONLY valid JSON with double quoted keys."
)


@dataclass
class PromptBundle:
    """Container for the system and user prompts."""

    system: str
    user: str

    def as_text(self) -> str:
        """Return a flattened text representation of the prompt."""

        return f"{self.system.strip()}\n\n{self.user.strip()}"


REASONING_EVAL_PROMPT = """
You are a meticulous and unbiased AI Interview Analyst 🧠. Your purpose is to evaluate a candidate's answer to a reasoning-based interview question using a detailed, multi-dimensional rubric.

**## Instructions:**
1.  **Review Inputs:** Carefully analyze the `QUESTION`, the `RESUME_CONTEXT`, and the `CANDIDATE_ANSWER`.
2.  **Consult Rubric:** Use the detailed `Rubric Definitions` below as your sole guide for scoring.
3.  **Analyze First:** Before assigning any scores, write a step-by-step analysis. For each dimension, critically assess the strengths and weaknesses of the answer against the rubric.
4.  **Score Second:** Based ONLY on your preceding analysis, assign a score from 1 (poor) to 5 (excellent) for each dimension.
5.  **Format Output:** Structure your complete analysis and scores into a single, valid JSON object as specified.

**## Inputs:**
* **QUESTION:** {question}
* **RESUME_CONTEXT:** {resume_context}
* **CANDIDATE_ANSWER:** {answer}

**## Rubric Definitions:**
* **Problem Comprehension:**
    * 1: Misunderstands the core question.
    * 3: Understands the main problem but misses key constraints or nuances.
    * 5: Demonstrates a deep, holistic understanding, perhaps re-framing the problem to reveal a key insight.
* **Structured Thinking:**
    * 1: Answer is chaotic and unstructured.
    * 3: Follows a basic structure but is somewhat disorganized or hard to follow.
    * 5: Employs a highly effective framework (e.g., STAR, SWOT), logically segmenting the problem and building a coherent case.
* **Identification of Assumptions:**
    * 1: Makes critical assumptions without acknowledging them.
    * 3: Acknowledges some, but not all, key assumptions.
    * 5: Explicitly calls out their important assumptions and explains why they are necessary.
* **Analysis of Trade-offs:**
    * 1: Presents a one-sided view with no consideration of alternatives.
    * 3: Mentions some pros and cons but analysis is superficial.
    * 5: Provides a balanced analysis of competing options, clearly weighing the trade-offs of their proposed solution.
* **Clarity of Communication:**
    * 1: Answer is confusing, uses excessive jargon, or is grammatically poor.
    * 3: Generally clear, but could be more concise or better organized.
    * 5: Communicates complex ideas clearly, concisely, and professionally.
* **Conclusion and Justification:**
    * 1: Fails to provide a conclusion or provides one with no support.
    * 3: Provides a reasonable conclusion but the justification is weak.
    * 5: Reaches a well-reasoned conclusion that is strongly supported by the preceding analysis and logic.

**## Output Format (JSON):**
{{
  "evaluation_type": "Reasoning",
  "overall_score": <Average of the 6 dimensional scores, rounded to two decimal places>,
  "dimensional_scores": {{
    "problem_comprehension": {{
      "analysis": "<Your step-by-step analysis for this dimension>",
      "score": <1-5>
    }},
    "structured_thinking": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "identification_of_assumptions": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "analysis_of_trade_offs": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "clarity_of_communication": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "conclusion_and_justification": {{
      "analysis": "<...>",
      "score": <1-5>
    }}
  }}
}}
"""


CONCEPTUAL_EVAL_PROMPT = """
You are a knowledgeable and precise AI Interview Analyst 📚. Your purpose is to evaluate a candidate's answer to a conceptual or factual interview question by comparing it against a ground truth and a detailed rubric.

**## Instructions:**
1.  **Review Inputs:** Carefully analyze the `QUESTION`, the `CANDIDATE_ANSWER`, and the `IDEAL_ANSWER_KEY_POINTS`.
2.  **Establish Ground Truth:** The `IDEAL_ANSWER_KEY_POINTS` is your source of truth. Base your evaluation of accuracy and depth directly on it.
3.  **Consult Rubric:** Use the detailed `Rubric Definitions` below as your sole guide for scoring.
4.  **Analyze First:** Before assigning any scores, write a step-by-step analysis. For each dimension, critically assess the candidate's answer against the rubric and the ideal answer.
5.  **Score Second:** Based ONLY on your preceding analysis, assign a score from 1 (poor) to 5 (excellent) for each dimension.
6.  **Format Output:** Structure your complete analysis and scores into a single, valid JSON object as specified.

**## Inputs:**
* **QUESTION:** {question}
* **IDEAL_ANSWER_KEY_POINTS:** {ideal_answer}
* **CANDIDATE_ANSWER:** {answer}

**## Rubric Definitions:**
* **Factual Accuracy:**
    * 1: Contains significant factual errors when compared to the ideal answer.
    * 3: Mostly accurate, but with minor inaccuracies or omissions.
    * 5: Completely aligns with the factual points in the ideal answer.
* **Depth of Knowledge:**
    * 1: Provides a very superficial, surface-level explanation.
    * 3: Explains the 'what' but not the 'why'; shows a partial understanding.
    * 5: Demonstrates a deep understanding of the topic, including its underlying principles and context, as reflected in the ideal answer.
* **Clarity of Explanation:**
    * 1: Answer is confusing, hard to follow, or poorly worded.
    * 3: Understandable, but could be clearer or more structured.
    * 5: Explains complex concepts in a clear, simple, and easy-to-understand way.
* **Practical Application:**
    * 1: Cannot connect the concept to any real-world examples.
    * 3: Provides a generic or weak example of the concept's application.
    * 5: Illustrates the concept with relevant, specific, and insightful real-world examples.
* **Handling of Nuance:**
    * 1: Presents the topic in an overly simplistic, black-and-white manner.
    * 3: Acknowledges some nuance, like exceptions or edge cases, but doesn't explore them.
    * 5: Discusses important nuances, trade-offs, or context where the concept applies differently.

**## Output Format (JSON):**
{{
  "evaluation_type": "Conceptual",
  "overall_score": <Average of the 5 dimensional scores, rounded to two decimal places>,
  "dimensional_scores": {{
    "factual_accuracy": {{
      "analysis": "<Your step-by-step analysis for this dimension>",
      "score": <1-5>
    }},
    "depth_of_knowledge": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "clarity_of_explanation": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "practical_application": {{
      "analysis": "<...>",
      "score": <1-5>
    }},
    "handling_of_nuance": {{
      "analysis": "<...>",
      "score": <1-5>
    }}
  }}
}}
"""


PROMPT_TEMPLATES: Dict[EvaluationType, str] = {
    EvaluationType.REASONING: REASONING_EVAL_PROMPT,
    EvaluationType.CONCEPTUAL: CONCEPTUAL_EVAL_PROMPT,
}

PromptFormatter = Callable[[EvaluationRequest], Dict[str, str]]


def _reasoning_prompt_kwargs(payload: EvaluationRequest) -> Dict[str, str]:
    return {
        "question": payload.question,
        "resume_context": payload.resume_context,
        "answer": payload.answer,
    }


def _conceptual_prompt_kwargs(payload: EvaluationRequest) -> Dict[str, str]:
    return {
        "question": payload.question,
        "ideal_answer": payload.ideal_answer_key_points,
        "answer": payload.answer,
    }


PROMPT_KWARGS: Dict[EvaluationType, PromptFormatter] = {
    EvaluationType.REASONING: _reasoning_prompt_kwargs,
    EvaluationType.CONCEPTUAL: _conceptual_prompt_kwargs,
}


class PromptEngine:
    """Create prompts tailored to the evaluation type."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    def build_prompt(self, payload: EvaluationRequest) -> PromptBundle:
        """Return the filled template for the incoming request."""

        template = PROMPT_TEMPLATES.get(payload.evaluation_type)
        formatter = PROMPT_KWARGS.get(payload.evaluation_type)
        if template is None or formatter is None:
            raise ValueError(f"Unsupported evaluation type: {payload.evaluation_type}")
        user_prompt = template.format(**formatter(payload))
        return PromptBundle(system=self._system_prompt, user=user_prompt)
