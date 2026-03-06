"""Shared fixtures and contract tests for retrieval strategies.

The ``RetrieverContractTests`` mixin defines the interface contract that
every Retriever implementation must satisfy.  BM25 and Vector test
modules inherit from it, supplying only a ``retriever`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.marley.retrieval import RetrievalResult, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "stpo-chunks.json"
FAQ_STPO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-stpo-chunks.json"
FAQ_AO_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "faq-ao-chunks.json"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

SMALL_CORPUS = [
    {"chunk_id": "c1", "text": "The master thesis has 30 credits.", "metadata": {"section": "par-23"}},
    {"chunk_id": "c2", "text": "Examination rules and grading policy.", "metadata": {"section": "par-10"}},
    {"chunk_id": "c3", "text": "Study abroad in the third semester.", "metadata": {"section": "par-5"}},
]


class RetrieverContractTests:
    """Mixin asserting the ``Retriever`` interface contract.

    Subclasses must provide a ``make_retriever`` method that returns a
    fresh, *unindexed* Retriever instance.
    """

    def make_retriever(self):
        raise NotImplementedError

    # -- index ---------------------------------------------------------

    def test_index_sets_size(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        assert r.size == 3

    def test_index_empty_corpus(self):
        r = self.make_retriever()
        r.index([])
        assert r.size == 0

    def test_reindex_replaces_corpus(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        assert r.size == 3
        r.index([{"chunk_id": "x", "text": "new doc", "metadata": {}}])
        assert r.size == 1

    # -- retrieve ------------------------------------------------------

    def test_retrieve_before_index_returns_empty(self):
        r = self.make_retriever()
        assert r.retrieve("test") == []

    def test_retrieve_returns_results(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("master thesis credits", k=2)
        assert 0 < len(results) <= 2

    def test_retrieve_returns_retrieval_result_type(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("thesis", k=1)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_retrieve_ranked_by_score(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("thesis credits", k=3)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top1_is_most_relevant(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("master thesis credits", k=1)
        assert results[0].chunk_id == "c1"

    def test_retrieve_respects_k(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("study", k=1)
        assert len(results) <= 1

    def test_metadata_preserved(self):
        r = self.make_retriever()
        r.index(SMALL_CORPUS)
        results = r.retrieve("master thesis credits", k=1)
        assert results[0].metadata["section"] == "par-23"
