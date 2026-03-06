"""Tests for the vector retrieval module."""

from pathlib import Path

import pytest

from src.marley.retrieval import RetrievalResult, VectorRetriever, load_chunks

from tests.retrieval.conftest import (
    FAQ_AO_CHUNKS_PATH,
    FAQ_STPO_CHUNKS_PATH,
    SMALL_CORPUS,
    STPO_CHUNKS_PATH,
    RetrieverContractTests,
)


# ---------------------------------------------------------------------------
# Unit tests — retriever contract + Vector-specific
# ---------------------------------------------------------------------------

class TestVectorRetrieverUnit(RetrieverContractTests):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self._persist_dir = tmp_path / "test_vectorstore"

    def make_retriever(self):
        return VectorRetriever(persist_directory=self._persist_dir)

    def test_score_range(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("master thesis", k=3)
        for res in results:
            assert -1.0 <= res.score <= 1.0

    def test_metadata_none_values_handled(self):
        corpus = [
            {"chunk_id": "c1", "text": "Some text about topics.",
             "metadata": {"parent": None, "page": 5}},
        ]
        r = self.make_retriever()
        r.index(corpus)
        results = r.retrieve("topics", k=1)
        assert results[0].metadata["parent"] == ""
        assert results[0].metadata["page"] == 5

    def test_metadata_list_values_handled(self):
        corpus = [
            {"chunk_id": "c1", "text": "Some text about topics.",
             "metadata": {"heading_path": ["Part I", "Section 1"]}},
        ]
        r = self.make_retriever()
        r.index(corpus)
        results = r.retrieve("topics", k=1)
        assert results[0].metadata["heading_path"] == "Part I > Section 1"

    def test_persistence_survives_new_instance(self):
        r1 = self.make_retriever()
        r1.index(SMALL_CORPUS)
        assert r1.size == 3
        r2 = VectorRetriever(persist_directory=self._persist_dir)
        assert r2.size == 3
        results = r2.retrieve("master thesis", k=1)
        assert len(results) > 0

    def test_size_zero_before_index(self):
        r = self.make_retriever()
        assert r.size == 0


# ---------------------------------------------------------------------------
# Integration tests (require chunk JSON files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not STPO_CHUNKS_PATH.exists(), reason="StPO chunks not available")
class TestVectorStPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self, tmp_path_factory):
        persist_dir = tmp_path_factory.mktemp("stpo_vectorstore")
        chunks = load_chunks(STPO_CHUNKS_PATH)
        r = VectorRetriever(persist_directory=persist_dir)
        r.index(chunks)
        return r

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
class TestVectorFAQStPOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self, tmp_path_factory):
        persist_dir = tmp_path_factory.mktemp("faq_stpo_vectorstore")
        chunks = load_chunks(FAQ_STPO_CHUNKS_PATH)
        r = VectorRetriever(persist_directory=persist_dir)
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
class TestVectorFAQAOIntegration:
    @pytest.fixture(scope="class")
    def retriever(self, tmp_path_factory):
        persist_dir = tmp_path_factory.mktemp("faq_ao_vectorstore")
        chunks = load_chunks(FAQ_AO_CHUNKS_PATH)
        r = VectorRetriever(persist_directory=persist_dir)
        r.index(chunks)
        return r

    def test_corpus_size(self, retriever):
        assert retriever.size == 50

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("application deadline", k=3)
        assert isinstance(results, list)
