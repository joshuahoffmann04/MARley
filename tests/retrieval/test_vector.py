"""Tests for the vector retrieval module."""

import shutil
from pathlib import Path

import pytest

from src.marley.retrieval import RetrievalResult, VectorRetriever, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "stpo-chunks.json"
FAQ_STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-stpo-chunks.json"
FAQ_AO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-ao-chunks.json"


# ---------------------------------------------------------------------------
# Unit tests (no external data required)
# ---------------------------------------------------------------------------

class TestVectorRetrieverUnit:
    """Unit tests using a small synthetic corpus and a temp directory."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self, tmp_path):
        self.persist_dir = tmp_path / "test_vectorstore"
        yield
        # Cleanup handled by tmp_path

    def _make_corpus(self):
        return [
            {
                "chunk_id": "c1",
                "text": "The master thesis has 30 credits.",
                "metadata": {"section": "par-23"},
            },
            {
                "chunk_id": "c2",
                "text": "Examination rules and grading policy.",
                "metadata": {"section": "par-10"},
            },
            {
                "chunk_id": "c3",
                "text": "Study abroad in the third semester.",
                "metadata": {"section": "par-5"},
            },
        ]

    def test_index_sets_size(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        assert r.size == 3

    def test_index_empty_corpus(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index([])
        assert r.size == 0

    def test_retrieve_before_index(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        assert r.retrieve("test") == []

    def test_retrieve_returns_results(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("master thesis credits", k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_retrieve_returns_retrieval_result_type(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("thesis", k=1)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_retrieve_ranked_by_score(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("thesis credits", k=3)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top1_is_most_relevant(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("master thesis credits", k=1)
        assert results[0].chunk_id == "c1"

    def test_retrieve_respects_k(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("study", k=1)
        assert len(results) <= 1

    def test_score_range(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("master thesis", k=3)
        for res in results:
            assert -1.0 <= res.score <= 1.0

    def test_reindex_replaces_corpus(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        assert r.size == 3
        r.index([{"chunk_id": "x", "text": "new doc", "metadata": {}}])
        assert r.size == 1

    def test_metadata_preserved(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(self._make_corpus())
        results = r.retrieve("master thesis credits", k=1)
        assert results[0].chunk_id == "c1"
        assert results[0].metadata["section"] == "par-23"

    def test_metadata_none_values_handled(self):
        corpus = [
            {
                "chunk_id": "c1",
                "text": "Some text about topics.",
                "metadata": {"parent": None, "page": 5},
            },
        ]
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(corpus)
        results = r.retrieve("topics", k=1)
        assert results[0].metadata["parent"] == ""
        assert results[0].metadata["page"] == 5

    def test_metadata_list_values_handled(self):
        corpus = [
            {
                "chunk_id": "c1",
                "text": "Some text about topics.",
                "metadata": {"heading_path": ["Part I", "Section 1"]},
            },
        ]
        r = VectorRetriever(persist_directory=self.persist_dir)
        r.index(corpus)
        results = r.retrieve("topics", k=1)
        assert results[0].metadata["heading_path"] == "Part I > Section 1"

    def test_persistence_survives_new_instance(self):
        r1 = VectorRetriever(persist_directory=self.persist_dir)
        r1.index(self._make_corpus())
        assert r1.size == 3

        # Create a new retriever pointing to the same directory
        r2 = VectorRetriever(persist_directory=self.persist_dir)
        assert r2.size == 3
        results = r2.retrieve("master thesis", k=1)
        assert len(results) > 0

    def test_size_property_zero_before_index(self):
        r = VectorRetriever(persist_directory=self.persist_dir)
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
