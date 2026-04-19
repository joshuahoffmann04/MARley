"""Tests for retrieval score normalization and threshold filtering."""

from __future__ import annotations

import pytest

from src.marley.models.retrieval import RetrievalResult
from src.marley.models.scoring import (
    compute_confidence,
    compute_fusion_confidence,
    filter_by_threshold,
    normalize_scores,
)


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, text="text", score=score, metadata={})


# ── BM25 normalization ─────────────────────────────────────────────────────


class TestNormalizeBM25:
    """Tests for BM25 saturation normalization: score / (score + k)."""

    def test_zero_score_maps_to_zero(self) -> None:
        results = normalize_scores([_result("c1", 0.0)], "bm25")
        assert results[0].score == 0.0

    def test_saturation_curve_default_k(self) -> None:
        """With k=1: score=1 -> 0.5, score=5 -> 5/6."""
        results = normalize_scores(
            [_result("c1", 1.0), _result("c2", 5.0)], "bm25",
        )
        assert results[0].score == pytest.approx(0.5)
        assert results[1].score == pytest.approx(5.0 / 6.0)

    def test_custom_k_parameter(self) -> None:
        """With k=10: score=10 -> 0.5."""
        results = normalize_scores([_result("c1", 10.0)], "bm25", bm25_k=10.0)
        assert results[0].score == pytest.approx(0.5)

    def test_empty_results(self) -> None:
        assert normalize_scores([], "bm25") == []

    def test_order_preserved(self) -> None:
        results = normalize_scores(
            [_result("c1", 10.0), _result("c2", 1.0), _result("c3", 5.0)],
            "bm25",
        )
        assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]


# ── Vector normalization ────────────────────────────────────────────────────


class TestNormalizeVector:
    """Tests for vector identity normalization."""

    def test_scores_unchanged(self) -> None:
        results = normalize_scores(
            [_result("c1", 0.8), _result("c2", 0.3)], "vector",
        )
        assert results[0].score == pytest.approx(0.8)
        assert results[1].score == pytest.approx(0.3)

    def test_empty_results(self) -> None:
        assert normalize_scores([], "vector") == []

    def test_metadata_preserved(self) -> None:
        r = RetrievalResult(chunk_id="c1", text="hello", score=0.9, metadata={"k": "v"})
        results = normalize_scores([r], "vector")
        assert results[0].metadata == {"k": "v"}
        assert results[0].text == "hello"


# ── RRF normalization ───────────────────────────────────────────────────────


class TestNormalizeRRF:
    """Tests for RRF normalization by theoretical maximum."""

    def test_theoretical_max_maps_to_one(self) -> None:
        """A document ranked #1 in all retrievers has max score n/(k+1)."""
        max_score = 2 / (1 + 1)  # 2 retrievers, DEFAULT_K_RRF=1
        results = normalize_scores([_result("c1", max_score)], "rrf")
        assert results[0].score == pytest.approx(1.0)

    def test_partial_score(self) -> None:
        max_score = 2 / (1 + 1)  # DEFAULT_K_RRF=1
        half = max_score / 2
        results = normalize_scores([_result("c1", half)], "rrf")
        assert results[0].score == pytest.approx(0.5)

    def test_custom_k_rrf(self) -> None:
        max_score = 3 / (10 + 1)  # 3 retrievers, k=10
        results = normalize_scores(
            [_result("c1", max_score)], "rrf",
            rrf_n_retrievers=3, rrf_k=10,
        )
        assert results[0].score == pytest.approx(1.0)

    def test_empty_results(self) -> None:
        assert normalize_scores([], "rrf") == []


# ── Invalid strategy ────────────────────────────────────────────────────────


class TestNormalizeInvalidStrategy:

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization strategy"):
            normalize_scores([_result("c1", 1.0)], "unknown")


# ── Threshold filtering ────────────────────────────────────────────────────


class TestFilterByThreshold:
    """Tests for threshold-based result filtering."""

    def test_filters_below_threshold(self) -> None:
        results = [_result("c1", 0.8), _result("c2", 0.2), _result("c3", 0.5)]
        filtered = filter_by_threshold(results, 0.5)
        assert [r.chunk_id for r in filtered] == ["c1", "c3"]

    def test_keeps_all_above(self) -> None:
        results = [_result("c1", 0.9), _result("c2", 0.7)]
        filtered = filter_by_threshold(results, 0.3)
        assert len(filtered) == 2

    def test_empty_after_filtering(self) -> None:
        results = [_result("c1", 0.1), _result("c2", 0.2)]
        filtered = filter_by_threshold(results, 0.5)
        assert filtered == []

    def test_exact_threshold_kept(self) -> None:
        results = [_result("c1", 0.5)]
        filtered = filter_by_threshold(results, 0.5)
        assert len(filtered) == 1


# ── Confidence computation ──────────────────────────────────────────────────


class TestComputeConfidence:
    """Tests for top-1 confidence computation."""

    def test_returns_max_score(self) -> None:
        results = [_result("c1", 0.3), _result("c2", 0.9), _result("c3", 0.5)]
        assert compute_confidence(results) == pytest.approx(0.9)

    def test_empty_results_returns_zero(self) -> None:
        assert compute_confidence([]) == 0.0

    def test_single_result(self) -> None:
        assert compute_confidence([_result("c1", 0.42)]) == pytest.approx(0.42)


# -- Fusion-aware confidence -----------------------------------------------


class TestComputeFusionConfidence:
    """Fusion-aware confidence from sub-retriever normalised scores.

    When a FusionRetriever wraps disjoint-corpus sub-retrievers, the raw
    RRF score at the fused level is mathematically constant. These tests
    lock in that ``compute_fusion_confidence`` works pre-fusion on the
    sub-retriever outputs and therefore varies with the actual query.
    """

    def test_empty_sub_results_returns_zero(self) -> None:
        assert compute_fusion_confidence([], "vector") == 0.0

    def test_all_empty_sub_results_returns_zero(self) -> None:
        assert compute_fusion_confidence([[], [], []], "vector") == 0.0

    def test_vector_strategy_uses_max_of_subretriever_tops(self) -> None:
        # Three sub-retrievers, top-1 scores 0.4 / 0.8 / 0.3.
        sub_results = [
            [_result("a1", 0.4), _result("a2", 0.3)],
            [_result("b1", 0.8), _result("b2", 0.5)],
            [_result("c1", 0.3), _result("c2", 0.1)],
        ]
        # Vector normalisation is identity, so the answer is plain max.
        assert compute_fusion_confidence(sub_results, "vector") == pytest.approx(0.8)

    def test_bm25_strategy_saturates_then_maxes(self) -> None:
        # Raw BM25 = 3.0, saturation with k=1 -> 3/(3+1) = 0.75. Biggest wins.
        sub_results = [
            [_result("a1", 1.0)],  # 1/2 = 0.5
            [_result("b1", 3.0)],  # 3/4 = 0.75
        ]
        conf = compute_fusion_confidence(sub_results, "bm25")
        assert conf == pytest.approx(0.75)

    def test_varies_per_query(self) -> None:
        """The whole point: different queries yield different confidences."""
        weak_query = [[_result("a1", 0.1)], [_result("b1", 0.2)]]
        strong_query = [[_result("a1", 0.9)], [_result("b1", 0.7)]]
        assert compute_fusion_confidence(weak_query, "vector") < compute_fusion_confidence(strong_query, "vector")

    def test_skips_empty_sub_retriever(self) -> None:
        # One empty, one with real results -> confidence from the non-empty one.
        sub_results = [[], [_result("b1", 0.6)]]
        assert compute_fusion_confidence(sub_results, "vector") == pytest.approx(0.6)
