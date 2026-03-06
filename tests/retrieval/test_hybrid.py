"""Tests for the hybrid retrieval module."""

from pathlib import Path

import pytest

from src.marley.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalResult,
    VectorRetriever,
    load_chunks,
)
from src.marley.retrieval.base import Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "stpo-chunks.json"
FAQ_STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-stpo-chunks.json"
FAQ_AO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-ao-chunks.json"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRetriever(Retriever):
    """Retriever that returns a fixed ranked list for testing."""

    def __init__(self, responses: dict[str, list[tuple[str, float]]] | None = None) -> None:
        self._responses = responses or {}
        self._size = 0

    def index(self, corpus: list[dict]) -> None:
        self._size = len(corpus)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        items = self._responses.get(query, [])[:k]
        return [
            RetrievalResult(chunk_id=cid, text=f"text-{cid}", score=score, metadata={})
            for cid, score in items
        ]

    @property
    def size(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Unit tests (no external data required)
# ---------------------------------------------------------------------------

class TestHybridRetrieverUnit:
    def test_requires_two_retrievers(self):
        r = _FakeRetriever()
        with pytest.raises(ValueError, match="Expected exactly 2 retrievers"):
            HybridRetriever(retrievers=(r,))

    def test_requires_two_retrievers_three(self):
        r = _FakeRetriever()
        with pytest.raises(ValueError, match="Expected exactly 2 retrievers"):
            HybridRetriever(retrievers=(r, r, r))

    def test_index_delegates_to_both(self):
        r1 = _FakeRetriever()
        r2 = _FakeRetriever()
        hybrid = HybridRetriever(retrievers=(r1, r2))
        corpus = [{"chunk_id": "c1", "text": "hello", "metadata": {}}]
        hybrid.index(corpus)
        assert r1.size == 1
        assert r2.size == 1

    def test_size_returns_first_retriever_size(self):
        r1 = _FakeRetriever()
        r2 = _FakeRetriever()
        hybrid = HybridRetriever(retrievers=(r1, r2))
        corpus = [{"chunk_id": "c1", "text": "hello", "metadata": {}}]
        hybrid.index(corpus)
        assert hybrid.size == 1

    def test_retrieve_before_index_returns_empty(self):
        r1 = _FakeRetriever()
        r2 = _FakeRetriever()
        hybrid = HybridRetriever(retrievers=(r1, r2))
        assert hybrid.retrieve("test") == []

    def test_retrieve_returns_retrieval_result_type(self):
        r1 = _FakeRetriever({"q": [("c1", 1.0)]})
        r2 = _FakeRetriever({"q": [("c1", 0.9)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_fuses_results(self):
        # c1 appears in both, c2 only in r1, c3 only in r2
        r1 = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5)]})
        r2 = _FakeRetriever({"q": [("c1", 0.9), ("c3", 0.4)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=3)
        ids = [r.chunk_id for r in results]
        # c1 should be first (appears in both)
        assert ids[0] == "c1"
        assert set(ids) == {"c1", "c2", "c3"}

    def test_rrf_scores_are_positive(self):
        r1 = _FakeRetriever({"q": [("c1", 1.0)]})
        r2 = _FakeRetriever({"q": [("c2", 0.9)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=2)
        assert all(r.score > 0 for r in results)

    def test_rrf_scores_sorted_descending(self):
        r1 = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5), ("c3", 0.2)]})
        r2 = _FakeRetriever({"q": [("c1", 0.9), ("c3", 0.4), ("c2", 0.1)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_k(self):
        r1 = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5)]})
        r2 = _FakeRetriever({"q": [("c1", 0.9), ("c3", 0.4)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=1)
        assert len(results) == 1

    def test_document_in_both_ranks_higher(self):
        # c1 only in r1 (rank 1), c2 in both (rank 2 in r1, rank 1 in r2)
        r1 = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5)]})
        r2 = _FakeRetriever({"q": [("c2", 0.9)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=2)
        # c2 appears in both retrievers -> higher RRF score
        assert results[0].chunk_id == "c2"

    def test_custom_k_rrf(self):
        r1 = _FakeRetriever({"q": [("c1", 1.0)]})
        r2 = _FakeRetriever({"q": [("c1", 0.9)]})
        hybrid_default = HybridRetriever(retrievers=(r1, r2), k_rrf=60)
        hybrid_custom = HybridRetriever(retrievers=(r1, r2), k_rrf=1)
        res_default = hybrid_default.retrieve("q", k=1)
        res_custom = hybrid_custom.retrieve("q", k=1)
        # With k_rrf=1: score = 2 * 1/(1+1) = 1.0
        # With k_rrf=60: score = 2 * 1/(60+1) ~ 0.033
        assert res_custom[0].score > res_default[0].score

    def test_reindex_replaces_corpus(self):
        r1 = _FakeRetriever()
        r2 = _FakeRetriever()
        hybrid = HybridRetriever(retrievers=(r1, r2))
        hybrid.index([{"chunk_id": "c1", "text": "a", "metadata": {}}])
        assert hybrid.size == 1
        hybrid.index([
            {"chunk_id": "c1", "text": "a", "metadata": {}},
            {"chunk_id": "c2", "text": "b", "metadata": {}},
        ])
        assert hybrid.size == 2

    def test_metadata_from_highest_scoring_source(self):
        r1 = _FakeRetriever()
        r2 = _FakeRetriever()
        r1._responses = {"q": [("c1", 0.5)]}
        r2._responses = {"q": [("c1", 0.9)]}
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=1)
        assert results[0].text == "text-c1"

    def test_no_duplicate_results(self):
        r1 = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5)]})
        r2 = _FakeRetriever({"q": [("c1", 0.9), ("c2", 0.4)]})
        hybrid = HybridRetriever(retrievers=(r1, r2))
        results = hybrid.retrieve("q", k=5)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Integration tests (require chunk JSON files + vectorstore)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not STPO_CHUNKS_PATH.exists() or not (VECTORSTORE_DIR / "stpo").exists(),
    reason="StPO chunks or vectorstore not available",
)
class TestHybridStPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(STPO_CHUNKS_PATH)
        bm25 = BM25Retriever()
        bm25.index(chunks)
        vector = VectorRetriever(persist_directory=VECTORSTORE_DIR / "stpo")
        return HybridRetriever(retrievers=(bm25, vector))

    def test_corpus_size(self, retriever):
        assert retriever.size > 100

    def test_thesis_query(self, retriever):
        results = retriever.retrieve("master thesis processing time", k=5)
        assert len(results) > 0
        chunk_ids = [r.chunk_id for r in results]
        assert any("par-23" in cid for cid in chunk_ids)

    def test_unique_results(self, retriever):
        results = retriever.retrieve("examination committee", k=10)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))


@pytest.mark.skipif(
    not FAQ_STPO_CHUNKS_PATH.exists() or not (VECTORSTORE_DIR / "faq-stpo").exists(),
    reason="FAQ-StPO chunks or vectorstore not available",
)
class TestHybridFAQStPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(FAQ_STPO_CHUNKS_PATH)
        bm25 = BM25Retriever()
        bm25.index(chunks)
        vector = VectorRetriever(persist_directory=VECTORSTORE_DIR / "faq-stpo")
        return HybridRetriever(retrievers=(bm25, vector))

    def test_corpus_size(self, retriever):
        assert retriever.size == 999

    def test_thesis_query(self, retriever):
        results = retriever.retrieve("How long is the master thesis?", k=5)
        assert len(results) > 0

    def test_results_have_faq_metadata(self, retriever):
        results = retriever.retrieve("credits", k=1)
        assert "faq_source" in results[0].metadata


@pytest.mark.skipif(
    not FAQ_AO_CHUNKS_PATH.exists() or not (VECTORSTORE_DIR / "faq-ao").exists(),
    reason="FAQ-AO chunks or vectorstore not available",
)
class TestHybridFAQAOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(FAQ_AO_CHUNKS_PATH)
        bm25 = BM25Retriever()
        bm25.index(chunks)
        vector = VectorRetriever(persist_directory=VECTORSTORE_DIR / "faq-ao")
        return HybridRetriever(retrievers=(bm25, vector))

    def test_corpus_size(self, retriever):
        assert retriever.size == 50

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("application deadline", k=3)
        assert isinstance(results, list)
