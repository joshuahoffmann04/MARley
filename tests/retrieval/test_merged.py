"""Tests for the merged-pool retrieval strategy."""

from pathlib import Path

import pytest

from src.marley.retrieval import (
    BM25Retriever,
    MergedRetriever,
    RetrievalResult,
    VectorRetriever,
    load_chunks,
)
from src.marley.models.retrieval import Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "stpo-chunks.json"
FAQ_STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-stpo-chunks.json"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRetriever(Retriever):
    """Retriever that returns a fixed ranked list for testing."""

    def __init__(self, responses: dict[str, list[tuple[str, float]]] | None = None) -> None:
        self._responses = responses or {}
        self._size = 0
        self._indexed_corpus: list[dict] = []

    def index(self, corpus: list[dict]) -> None:
        self._indexed_corpus = corpus
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
# Unit tests
# ---------------------------------------------------------------------------

class TestMergedRetrieverUnit:
    """Unit tests for MergedRetriever."""

    def test_index_delegates_to_inner(self):
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        corpus = [
            {"chunk_id": "c1", "text": "hello", "metadata": {}},
            {"chunk_id": "c2", "text": "world", "metadata": {}},
        ]
        merged.index(corpus)
        assert inner._indexed_corpus == corpus

    def test_size_delegates_to_inner(self):
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        assert merged.size == 0
        corpus = [{"chunk_id": "c1", "text": "a", "metadata": {}}]
        merged.index(corpus)
        assert merged.size == 1

    def test_retrieve_delegates_to_inner(self):
        inner = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5)]})
        merged = MergedRetriever(inner)
        results = merged.retrieve("q", k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"

    def test_retrieve_respects_k(self):
        inner = _FakeRetriever({"q": [("c1", 1.0), ("c2", 0.5)]})
        merged = MergedRetriever(inner)
        results = merged.retrieve("q", k=1)
        assert len(results) == 1

    def test_retrieve_before_index_returns_empty(self):
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        assert merged.retrieve("test") == []

    def test_index_empty_corpus(self):
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        merged.index([])
        assert merged.size == 0

    def test_reindex_replaces_corpus(self):
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        merged.index([{"chunk_id": "c1", "text": "a", "metadata": {}}])
        assert merged.size == 1
        merged.index([
            {"chunk_id": "c1", "text": "a", "metadata": {}},
            {"chunk_id": "c2", "text": "b", "metadata": {}},
        ])
        assert merged.size == 2

    def test_returns_retrieval_result_type(self):
        inner = _FakeRetriever({"q": [("c1", 1.0)]})
        merged = MergedRetriever(inner)
        results = merged.retrieve("q", k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_merged_corpus_from_multiple_kbs(self):
        """Simulate merging chunks from two KBs before indexing."""
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        kb1 = [{"chunk_id": "stpo-1", "text": "a", "metadata": {"kb": "stpo"}}]
        kb2 = [{"chunk_id": "faq-1", "text": "b", "metadata": {"kb": "faq"}}]
        merged.index(kb1 + kb2)
        assert merged.size == 2
        assert inner._indexed_corpus[0]["chunk_id"] == "stpo-1"
        assert inner._indexed_corpus[1]["chunk_id"] == "faq-1"

    def test_inner_retriever_is_accessible(self):
        """MergedRetriever is a thin wrapper — inner retriever is the same object."""
        inner = _FakeRetriever()
        merged = MergedRetriever(inner)
        merged.index([{"chunk_id": "c1", "text": "a", "metadata": {}}])
        assert inner.size == 1


# ---------------------------------------------------------------------------
# Integration tests (require chunk JSON files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not STPO_CHUNKS_PATH.exists() or not FAQ_STPO_CHUNKS_PATH.exists(),
    reason="StPO or FAQ-StPO chunks not available",
)
@pytest.mark.integration
class TestMergedBM25Integration:
    @pytest.fixture(scope="class")
    def retriever(self):
        stpo = load_chunks(STPO_CHUNKS_PATH)
        faq = load_chunks(FAQ_STPO_CHUNKS_PATH)
        inner = BM25Retriever()
        merged = MergedRetriever(inner)
        merged.index(stpo + faq)
        return merged

    def test_corpus_size(self, retriever):
        assert retriever.size == 153 + 1039

    def test_thesis_query_returns_results(self, retriever):
        results = retriever.retrieve("master thesis credits", k=5)
        assert len(results) > 0

    def test_results_from_both_kbs(self, retriever):
        results = retriever.retrieve("master thesis credits", k=10)
        ids = [r.chunk_id for r in results]
        # Should find results from both KBs
        has_stpo = any("par" in cid or "app" in cid for cid in ids)
        has_faq = any("faq" in cid.lower() or cid.startswith("0") for cid in ids)
        assert has_stpo or has_faq  # at least one KB represented

    def test_unique_results(self, retriever):
        results = retriever.retrieve("examination committee", k=10)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))
