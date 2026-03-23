"""Tests for the abstention-aware pipeline orchestrator."""

from __future__ import annotations

import pytest

from src.marley.abstention.pipeline import run_with_abstention
from src.marley.models.retrieval import RetrievalResult
from tests.conftest import FixedRetriever, StubGenerator


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, text=f"text-{chunk_id}", score=score, metadata={})


# ── Tests ───────────────────────────────────────────────────────────────────


class TestRunWithAbstention:
    """Tests for the run_with_abstention orchestrator."""

    def test_level1_abstention_low_scores(self) -> None:
        """All scores below threshold -> Level 1 abstention."""
        retriever = FixedRetriever([_result("c1", 0.1), _result("c2", 0.05)])
        generator = StubGenerator(answer="should not be called")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.3, normalization_strategy="vector",
        )
        assert result.abstained is True
        assert result.level == 1
        assert "confidence below threshold" in result.reason
        assert result.answer == ""

    def test_level2_abstention_llm_abstains(self) -> None:
        """LLM returns ABSTENTION: -> Level 2 abstention."""
        retriever = FixedRetriever([_result("c1", 0.8)])
        generator = StubGenerator(answer="ABSTENTION: context lacks info")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.3, normalization_strategy="vector",
        )
        assert result.abstained is True
        assert result.level == 2
        assert result.reason == "context lacks info"
        assert result.answer == ""
        assert result.model == "stub-model"

    def test_normal_answer(self) -> None:
        """Both levels pass -> normal answer returned."""
        retriever = FixedRetriever([_result("c1", 0.9), _result("c2", 0.7)])
        generator = StubGenerator(answer="The study period is four semesters.")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.3, normalization_strategy="vector",
        )
        assert result.abstained is False
        assert result.level is None
        assert result.reason == ""
        assert result.answer == "The study period is four semesters."

    def test_confidence_computed_correctly(self) -> None:
        """Confidence should be the top-1 normalized score."""
        retriever = FixedRetriever([_result("c1", 0.8), _result("c2", 0.3)])
        generator = StubGenerator(answer="Answer text")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.1, normalization_strategy="vector",
        )
        assert result.confidence == pytest.approx(0.8)

    def test_filtered_results_passed_to_generator(self) -> None:
        """Only chunks above threshold should be in retrieval_results."""
        retriever = FixedRetriever([_result("c1", 0.8), _result("c2", 0.1)])
        generator = StubGenerator(answer="Answer")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.5, normalization_strategy="vector",
        )
        assert len(result.retrieval_results) == 1
        assert result.retrieval_results[0]["chunk_id"] == "c1"

    def test_threshold_zero_no_filtering(self) -> None:
        """Threshold=0 keeps all results."""
        retriever = FixedRetriever([_result("c1", 0.01), _result("c2", 0.001)])
        generator = StubGenerator(answer="Answer")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.0, normalization_strategy="vector",
        )
        assert result.abstained is False
        assert len(result.retrieval_results) == 2

    def test_threshold_one_always_abstains(self) -> None:
        """Threshold=1.0 filters everything (unless score is exactly 1.0)."""
        retriever = FixedRetriever([_result("c1", 0.99)])
        generator = StubGenerator(answer="should not be called")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=1.0, normalization_strategy="vector",
        )
        assert result.abstained is True
        assert result.level == 1

    def test_bm25_normalization_strategy(self) -> None:
        """BM25 normalization transforms scores before threshold check."""
        # BM25 raw score=1.0 with k=1.0 -> normalized=0.5
        retriever = FixedRetriever([_result("c1", 1.0)])
        generator = StubGenerator(answer="Answer")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.3, normalization_strategy="bm25",
            normalization_params={"bm25_k": 1.0},
        )
        assert result.abstained is False
        assert result.confidence == pytest.approx(0.5)

    def test_empty_retrieval_results(self) -> None:
        """No retrieval results -> Level 1 abstention."""
        retriever = FixedRetriever([])
        generator = StubGenerator(answer="should not be called")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.3, normalization_strategy="vector",
        )
        assert result.abstained is True
        assert result.level == 1
        assert result.confidence == 0.0

    def test_result_fields_complete(self) -> None:
        """All AbstentionResult fields should be populated."""
        retriever = FixedRetriever([_result("c1", 0.7)])
        generator = StubGenerator(answer="Answer text", model="test-llm")
        result = run_with_abstention(
            "query", retriever, generator,
            threshold=0.3, normalization_strategy="vector",
        )
        assert result.abstained is False
        assert result.level is None
        assert result.reason == ""
        assert result.answer == "Answer text"
        assert result.confidence == pytest.approx(0.7)
        assert result.model == "test-llm"
        assert len(result.retrieval_results) == 1
