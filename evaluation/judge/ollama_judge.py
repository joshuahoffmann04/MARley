"""Ollama-backed LLM judge for the MARley evaluation framework.

Uses a locally hosted Ollama model to evaluate generated answers on
faithfulness, answer relevance, and correctness. JSON-mode output
ensures structured, parseable responses.
"""

from __future__ import annotations

import json
import re

import ollama as ollama_lib

from evaluation.judge.base import Judge, JudgementResult
from evaluation.judge.prompts import build_judge_messages

_ABSTENTION_PREFIX = "ABSTENTION:"
_SCORE_KEYS = ("faithfulness", "answer_relevance", "correctness")


class OllamaJudge(Judge):
    """Evaluate generated answers using a locally hosted Ollama model.

    Sends a single structured prompt requesting JSON scores for all three
    criteria. Falls back to score 0.0 when the model output cannot be
    parsed.
    """

    def __init__(
        self,
        model: str = "llama3.1:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._client = ollama_lib.Client(host=base_url)

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
        """Evaluate a generated answer using the Ollama LLM judge.

        For abstained answers (empty or starting with 'ABSTENTION:'), the
        function returns a sentinel result: faithfulness=1.0 (no false
        claims were made), answer_relevance=0.0, correctness=0.0.

        Args:
            question_id: Identifier used to correlate results.
            question: The original question text.
            context: Retrieval context chunks.
            generated_answer: The answer produced by the pipeline.
            reference_answer: The ground-truth reference answer.

        Returns:
            JudgementResult with scores in [0.0, 1.0].
        """
        # Handle abstentions — no answer to evaluate
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

        response = self._client.chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        raw = response.message.content.strip()
        scores = _parse_scores(raw)

        return JudgementResult(
            question_id=question_id,
            faithfulness=scores["faithfulness"],
            answer_relevance=scores["answer_relevance"],
            correctness=scores["correctness"],
            model=response.model or self.model,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_scores(raw: str) -> dict[str, float]:
    """Parse LLM JSON output into a score dict.

    Tries strict JSON parsing first; falls back to regex extraction for
    models that wrap the JSON in prose. Returns 0.0 for any key that
    cannot be extracted or is out of range.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, RecursionError, ValueError):
        # Try to find a flat JSON object anywhere in the output.
        # RecursionError can occur when the model returns deeply nested JSON.
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except (json.JSONDecodeError, RecursionError, ValueError):
                data = {}
        else:
            data = {}

    result: dict[str, float] = {}
    for key in _SCORE_KEYS:
        raw_val = data.get(key, 0.0)
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = 0.0
        result[key] = max(0.0, min(1.0, val))
    return result
