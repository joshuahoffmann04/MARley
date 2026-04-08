"""LLM judge base classes for the MARley evaluation framework.

Defines the JudgementResult dataclass and the Judge abstract base class.
All judge implementations inherit from Judge to ensure a consistent
interface across Ollama and OpenAI backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgementResult:
    """Scores assigned by an LLM judge for a single question.

    All scores are in [0.0, 1.0]. A score of -1.0 indicates that the
    criterion could not be evaluated (e.g., faithfulness for an abstained
    answer with empty context).
    """

    question_id: str
    faithfulness: float
    """Does the answer only use information present in the context?
    1.0 = fully grounded, 0.0 = contains hallucinations."""

    answer_relevance: float
    """Does the answer address the question asked?
    1.0 = fully relevant, 0.0 = completely off-topic."""

    correctness: float
    """Does the answer match the reference answer?
    1.0 = fully correct and complete, 0.0 = completely wrong."""

    model: str
    """Identifier of the judge model."""


class Judge(ABC):
    """Abstract base for all LLM judge implementations."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the underlying judge model."""

    @abstractmethod
    def judge(
        self,
        question_id: str,
        question: str,
        context: list[dict],
        generated_answer: str,
        reference_answer: str,
    ) -> JudgementResult:
        """Evaluate a generated answer on three criteria.

        Args:
            question_id: Identifier of the question being evaluated.
            question: The original question text.
            context: List of context chunk dicts (with 'chunk_id' and 'text').
            generated_answer: The answer produced by the generator.
            reference_answer: The ground-truth reference answer.

        Returns:
            A JudgementResult with faithfulness, answer_relevance,
            and correctness scores.
        """
