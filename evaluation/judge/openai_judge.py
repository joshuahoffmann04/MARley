"""OpenAI-backed LLM judge for the MARley evaluation framework.

Uses the OpenAI API to evaluate generated answers on faithfulness,
answer relevance, and correctness. Requires the OPENAI_API_KEY
environment variable to be set.

This module is a production-ready stub: it implements the full Judge
interface but requires an active OpenAI subscription. The default model
is gpt-4o-mini, which balances evaluation quality with cost.
"""

from __future__ import annotations

import json
import os
import re

from evaluation.judge.base import Judge, JudgementResult
from evaluation.judge.prompts import build_judge_messages

_ABSTENTION_PREFIX = "ABSTENTION:"
_SCORE_KEYS = ("faithfulness", "answer_relevance", "correctness")
_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIJudge(Judge):
    """Evaluate generated answers using the OpenAI API.

    Requires the ``openai`` package (``pip install openai``) and the
    ``OPENAI_API_KEY`` environment variable. The API key can also be
    supplied via the constructor.

    Example:
        >>> judge = OpenAIJudge(model="gpt-4o-mini")
        >>> result = judge.judge(
        ...     question_id="eval-001",
        ...     question="How long is the study period?",
        ...     context=[{"text": "The standard study period is 4 semesters."}],
        ...     generated_answer="The study period is 4 semesters.",
        ...     reference_answer="4 semesters.",
        ... )
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        """Initialise the OpenAI judge.

        Args:
            model: OpenAI model identifier (default: gpt-4o-mini).
            api_key: OpenAI API key. Falls back to the OPENAI_API_KEY
                environment variable if not supplied.

        Raises:
            ImportError: If the ``openai`` package is not installed.
            ValueError: If no API key is available.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIJudge. "
                "Install it with: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key provided. Pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )

        self._model = model
        self._client = openai.OpenAI(api_key=resolved_key)

    @property
    def model(self) -> str:
        return self._model

    def judge(
        self,
        question_id: str,
        question: str,
        context: list[dict],
        generated_answer: str,
        reference_answer: str,
    ) -> JudgementResult:
        """Evaluate a generated answer using the OpenAI API.

        Sends a structured JSON prompt and parses the three evaluation
        scores from the response. Abstained answers receive a sentinel
        result (faithfulness=1.0, relevance=0.0, correctness=0.0).

        Args:
            question_id: Identifier used to correlate results.
            question: The original question text.
            context: Retrieval context chunks.
            generated_answer: The answer produced by the pipeline.
            reference_answer: The ground-truth reference answer.

        Returns:
            JudgementResult with scores in [0.0, 1.0].
        """
        if (
            not generated_answer
            or generated_answer.strip().upper().startswith(_ABSTENTION_PREFIX)
        ):
            return JudgementResult(
                question_id=question_id,
                faithfulness=1.0,
                answer_relevance=0.0,
                correctness=0.0,
                model=self.model,
            )

        messages = build_judge_messages(
            question, context, generated_answer, reference_answer
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        scores = _parse_scores(raw)

        return JudgementResult(
            question_id=question_id,
            faithfulness=scores["faithfulness"],
            answer_relevance=scores["answer_relevance"],
            correctness=scores["correctness"],
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_scores(raw: str) -> dict[str, float]:
    """Parse LLM JSON output into a score dict."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    result: dict[str, float] = {}
    for key in _SCORE_KEYS:
        raw_val = data.get(key, 0.0)
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = 0.0
        result[key] = max(0.0, min(1.0, val))
    return result
