"""Tests for the BM25 retrieval module."""

import pytest

from src.marley.retrieval import BM25Retriever, load_chunks
from src.marley.retrieval.bm25 import _tokenize

from tests.retrieval.conftest import (
    FAQ_AO_CHUNKS_PATH,
    FAQ_STPO_CHUNKS_PATH,
    STPO_CHUNKS_PATH,
    RetrieverContractTests,
)


# ---------------------------------------------------------------------------
# Unit tests — tokenizer
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

    def test_preserves_german_umlauts(self):
        tokens = _tokenize("Prüfungsordnung für Übungen")
        assert "prüfungsordnung" in tokens
        assert "für" in tokens
        assert "übungen" in tokens


# ---------------------------------------------------------------------------
# Unit tests — retriever contract + BM25-specific
# ---------------------------------------------------------------------------


class TestBM25RetrieverUnit(RetrieverContractTests):
    def make_retriever(self):
        return BM25Retriever()

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

    def test_index_validates_corpus(self):
        r = BM25Retriever()
        with pytest.raises(ValueError, match="missing required keys"):
            r.index([{"chunk_id": "c1", "text": "hello"}])  # missing metadata


# ---------------------------------------------------------------------------
# Integration tests (require chunk JSON files)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not STPO_CHUNKS_PATH.exists(), reason="StPO chunks not available")
@pytest.mark.integration
class TestBM25StPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(STPO_CHUNKS_PATH)
        r = BM25Retriever()
        r.index(chunks)
        return r

    def test_corpus_size(self, retriever):
        assert retriever.size == 153

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
    not FAQ_STPO_CHUNKS_PATH.exists(), reason="FAQ-StPO chunks not available"
)
@pytest.mark.integration
class TestBM25FAQStPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(FAQ_STPO_CHUNKS_PATH)
        r = BM25Retriever()
        r.index(chunks)
        return r

    def test_corpus_size(self, retriever):
        assert retriever.size == 1039

    def test_thesis_query(self, retriever):
        results = retriever.retrieve("How long is the master thesis?", k=5)
        assert len(results) > 0

    def test_results_have_faq_metadata(self, retriever):
        results = retriever.retrieve("credits", k=1)
        assert "faq_source" in results[0].metadata


@pytest.mark.skipif(
    not FAQ_AO_CHUNKS_PATH.exists(), reason="FAQ-AO chunks not available"
)
@pytest.mark.integration
class TestBM25FAQAOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        chunks = load_chunks(FAQ_AO_CHUNKS_PATH)
        r = BM25Retriever()
        r.index(chunks)
        return r

    def test_corpus_size_placeholder(self, retriever):
        assert retriever.size == 0

    def test_retrieve_empty_corpus(self, retriever):
        results = retriever.retrieve("application deadline", k=3)
        assert results == []
