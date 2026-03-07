"""LLM-as-Judge for generation evaluation.

Provides a cleanly decoupled Judge interface so the evaluation strategy
can be replaced without modifying the evaluation runner. The default
implementation uses an Ollama model to assess semantic correctness.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ollama as ollama_lib

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. Your task is to determine whether "
    "a generated answer is semantically correct by comparing it to a "
    "reference answer.\n\n"
    "Rules:\n"
    "- The generated answer is CORRECT if it conveys the same key "
    "information as the reference, even if worded differently.\n"
    "- Minor differences in phrasing or additional non-contradictory "
    "details are acceptable.\n"
    "- The generated answer is INCORRECT if it contains factual errors, "
    "hallucinations, or misses the core information.\n\n"
    "Respond with ONLY a JSON object (no markdown, no extra text):\n"
    '{"correct": true/false, "confidence": 0.0-1.0, "reasoning": "..."}'
)


@dataclass
class JudgementResult:
    """Result of a single judge evaluation."""

    correct: bool
    confidence: float
    reasoning: str


class Judge(ABC):
    """Abstract base for answer correctness judges."""

    @abstractmethod
    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
    ) -> JudgementResult:
        """Judge whether the generated answer is correct."""


def _parse_judgement(text: str) -> JudgementResult:
    """Parse a JSON judgement from the LLM response.

    Handles common formatting issues like markdown code fences.
    Falls back to keyword-based heuristic if JSON parsing fails.
    """
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        return JudgementResult(
            correct=bool(data.get("correct", False)),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: check if the response contains clear indicators
        lower = text.lower()
        if '"correct": true' in lower or '"correct":true' in lower:
            return JudgementResult(correct=True, confidence=0.5, reasoning=text)
        return JudgementResult(correct=False, confidence=0.5, reasoning=text)


class OllamaJudge(Judge):
    """Judge answer correctness using an Ollama-hosted LLM."""

    def __init__(
        self,
        model: str = "llama3.1:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._client = ollama_lib.Client(host=base_url)

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
    ) -> JudgementResult:
        """Evaluate correctness by prompting the LLM as a judge."""
        user_content = (
            f"Question: {question}\n\n"
            f"Reference Answer: {reference_answer}\n\n"
            f"Generated Answer: {generated_answer}"
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self._client.chat(
            model=self.model,
            messages=messages,
            format="json",
        )

        return _parse_judgement(response.message.content)
