"""Tests for the BM25 retrieval module."""

from pathlib import Path

import pytest

from src.marley.retrieval import BM25Retriever, RetrievalResult, load_chunks
from src.marley.retrieval.bm25 import _tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "stpo-chunks.json"
FAQ_STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-stpo-chunks.json"
FAQ_AO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-ao-chunks.json"


# ---------------------------------------------------------------------------
# Unit tests (no external data required)
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_splits_whitespace(self):
        assert _tokenize("a  b\tc\nd") == ["a", "b", "c", "d"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_preserves_punctuation(self):
        tokens = _tokenize("What is §23?")
        assert "§23?" in tokens


class TestBM25RetrieverUnit:
    def _make_corpus(self):
        return [
            {"chunk_id": "c1", "text": "The master thesis has 30 credits.", "metadata": {}},
            {"chunk_id": "c2", "text": "Examination rules and grading policy.", "metadata": {}},
            {"chunk_id": "c3", "text": "Study abroad in the third semester.", "metadata": {}},
        ]

    def test_index_sets_size(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        assert r.size == 3

    def test_index_empty_corpus(self):
        r = BM25Retriever()
        r.index([])
        assert r.size == 0

    def test_retrieve_before_index(self):
        r = BM25Retriever()
        assert r.retrieve("test") == []

    def test_retrieve_returns_results(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        results = r.retrieve("master thesis credits", k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_retrieve_returns_retrieval_result_type(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        results = r.retrieve("thesis", k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_ranked_by_score(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        results = r.retrieve("thesis credits", k=3)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top1_is_most_relevant(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        results = r.retrieve("master thesis credits", k=1)
        assert results[0].chunk_id == "c1"

    def test_retrieve_respects_k(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        results = r.retrieve("study", k=1)
        assert len(results) <= 1

    def test_retrieve_filters_zero_scores(self):
        r = BM25Retriever()
        corpus = [
            {"chunk_id": "c1", "text": "alpha beta gamma", "metadata": {}},
            {"chunk_id": "c2", "text": "delta epsilon zeta", "metadata": {}},
        ]
        r.index(corpus)
        results = r.retrieve("alpha", k=5)
        for res in results:
            assert res.score > 0

    def test_reindex_replaces_corpus(self):
        r = BM25Retriever()
        r.index(self._make_corpus())
        assert r.size == 3
        r.index([{"chunk_id": "x", "text": "new doc", "metadata": {}}])
        assert r.size == 1

    def test_metadata_preserved(self):
        r = BM25Retriever()
        corpus = [
            {"chunk_id": "c1", "text": "hello world greeting", "metadata": {"section": "s1"}},
            {"chunk_id": "c2", "text": "foo bar baz", "metadata": {"section": "s2"}},
            {"chunk_id": "c3", "text": "delta epsilon zeta", "metadata": {"section": "s3"}},
        ]
        r.index(corpus)
        results = r.retrieve("hello world", k=1)
        assert results[0].chunk_id == "c1"
        assert results[0].metadata == {"section": "s1"}


# ---------------------------------------------------------------------------
# Integration tests (require chunk JSON files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not STPO_CHUNKS_PATH.exists(), reason="StPO chunks not available")
class TestBM25StPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(STPO_CHUNKS_PATH)
        r = BM25Retriever()
        r.index(chunks)
        return r

    @pytest.fixture(scope="class")
    def chunks(self):
        return load_chunks(STPO_CHUNKS_PATH)

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


@pytest.mark.skipif(not FAQ_STPO_CHUNKS_PATH.exists(), reason="FAQ-StPO chunks not available")
class TestBM25FAQStPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(FAQ_STPO_CHUNKS_PATH)
        r = BM25Retriever()
        r.index(chunks)
        return r

    def test_corpus_size(self, retriever):
        assert retriever.size == 999

    def test_thesis_query(self, retriever):
        results = retriever.retrieve("How long is the master thesis?", k=5)
        assert len(results) > 0

    def test_results_have_faq_metadata(self, retriever):
        results = retriever.retrieve("credits", k=1)
        assert "faq_source" in results[0].metadata


@pytest.mark.skipif(not FAQ_AO_CHUNKS_PATH.exists(), reason="FAQ-AO chunks not available")
class TestBM25FAQAOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(FAQ_AO_CHUNKS_PATH)
        r = BM25Retriever()
        r.index(chunks)
        return r

    def test_corpus_size(self, retriever):
        assert retriever.size == 50

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("application deadline", k=3)
        assert isinstance(results, list)
