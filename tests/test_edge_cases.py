"""Edge-case tests that guard pipeline components against empty or
extreme inputs. Derived from the pre-thesis audit (F-2-06)."""

from __future__ import annotations

import pytest

from src.marley.models.retrieval import RetrievalResult
from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)
from src.marley.retrieval.bm25 import BM25Retriever


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------


class TestNormalizeScoresEdgeCases:
    def test_empty_input_returns_empty(self) -> None:
        assert normalize_scores([], "vector") == []
        assert normalize_scores([], "bm25", bm25_k=1.0) == []
        assert normalize_scores([], "rrf", rrf_n_retrievers=2, rrf_k=1) == []

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_scores([], "not-a-strategy")


class TestComputeConfidenceEdgeCases:
    def test_empty_returns_zero(self) -> None:
        assert compute_confidence([]) == 0.0

    def test_single_result_returns_score(self) -> None:
        r = RetrievalResult(chunk_id="c1", text="t", score=0.42, metadata={})
        assert compute_confidence([r]) == pytest.approx(0.42)

    def test_max_score_is_returned(self) -> None:
        rs = [
            RetrievalResult(chunk_id="c1", text="t", score=0.2, metadata={}),
            RetrievalResult(chunk_id="c2", text="t", score=0.9, metadata={}),
            RetrievalResult(chunk_id="c3", text="t", score=0.5, metadata={}),
        ]
        assert compute_confidence(rs) == pytest.approx(0.9)


class TestFilterByThresholdEdgeCases:
    def test_threshold_zero_keeps_all(self) -> None:
        rs = [RetrievalResult(chunk_id=f"c{i}", text="t", score=i * 0.1, metadata={}) for i in range(5)]
        assert len(filter_by_threshold(rs, 0.0)) == 5

    def test_threshold_one_drops_all_below_one(self) -> None:
        rs = [RetrievalResult(chunk_id="c1", text="t", score=0.999, metadata={})]
        assert filter_by_threshold(rs, 1.0) == []

    def test_empty_input_returns_empty(self) -> None:
        assert filter_by_threshold([], 0.5) == []


# ---------------------------------------------------------------------------
# BM25 edge cases
# ---------------------------------------------------------------------------


class TestBM25EdgeCases:
    def test_empty_index_returns_empty(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([])
        assert bm25.retrieve("any query", k=5) == []

    def test_empty_query_returns_empty(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([
            {"chunk_id": "c1", "text": "something here", "metadata": {}},
        ])
        assert bm25.retrieve("", k=5) == []
        assert bm25.retrieve("    ", k=5) == []

    def test_k_zero_returns_empty(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([
            {"chunk_id": "c1", "text": "thesis credits", "metadata": {}},
        ])
        assert bm25.retrieve("thesis", k=0) == []

    def test_k_larger_than_corpus_returns_corpus_size_at_most(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([
            {"chunk_id": "c1", "text": "thesis credits module", "metadata": {}},
            {"chunk_id": "c2", "text": "thesis defence ceremony", "metadata": {}},
        ])
        out = bm25.retrieve("thesis", k=1000)
        assert len(out) <= 2
        assert all(r.score > 0 for r in out)
