"""Shared test fixtures and stub implementations.

Provides canonical stub classes for Retriever and Generator that
replace the per-file duplicates.  All test modules import from here
to ensure a single source of truth.
"""

from __future__ import annotations

import pytest

from src.marley.models.generation import Generator, GenerationResult
from src.marley.models.retrieval import RetrievalResult, Retriever


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SMALL_CORPUS = [
    {"chunk_id": "c1", "text": "The master thesis has 30 credits.", "metadata": {"section": "par-23"}},
    {"chunk_id": "c2", "text": "Examination rules and grading policy.", "metadata": {"section": "par-10"}},
    {"chunk_id": "c3", "text": "Study abroad in the third semester.", "metadata": {"section": "par-5"}},
]


# ---------------------------------------------------------------------------
# Stub retrievers
# ---------------------------------------------------------------------------


class KeywordRetriever(Retriever):
    """Retriever scoring by keyword overlap.

    Scores each document by counting query-word overlap and multiplying
    by a configurable factor.  Useful when tests need realistic
    BM25-style ranking without external dependencies.
    """

    def __init__(self, score_multiplier: float = 0.5) -> None:
        self._corpus: list[dict] = []
        self._score_multiplier = score_multiplier

    def index(self, corpus: list[dict]) -> None:
        self._corpus = corpus

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        query_words = set(query.lower().split())
        scored = []
        for doc in self._corpus:
            words = set(doc["text"].lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                scored.append((doc, overlap * self._score_multiplier))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for doc, score in scored[:k]
        ]

    @property
    def size(self) -> int:
        return len(self._corpus)


class FixedRetriever(Retriever):
    """Retriever returning pre-configured results.

    Used when tests need exact control over retrieval output
    without any scoring logic.
    """

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results or []

    def index(self, corpus: list[dict]) -> None:
        pass

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return self._results[:k]

    @property
    def size(self) -> int:
        return len(self._results)


# ---------------------------------------------------------------------------
# Stub generator
# ---------------------------------------------------------------------------


class StubGenerator(Generator):
    """Generator returning a fixed answer or abstaining on keywords.

    Configurable answer text, model name, and abstention keywords.
    Covers all generator stub use cases across the test suite.
    """

    def __init__(
        self,
        answer: str = "The answer is 42.",
        model: str = "stub-model",
        abstain_keywords: set[str] | None = None,
    ) -> None:
        self._answer = answer
        self.model = model
        self._abstain_keywords = abstain_keywords or set()

    def generate(self, query: str, context: list[dict]) -> GenerationResult:
        for kw in self._abstain_keywords:
            if kw in query.lower():
                return GenerationResult(
                    answer="ABSTENTION: insufficient context",
                    model=self.model,
                )
        return GenerationResult(
            answer=self._answer,
            model=self.model,
            context_chunk_ids=[c["chunk_id"] for c in context if "chunk_id" in c],
            prompt_tokens=10,
            completion_tokens=5,
        )
