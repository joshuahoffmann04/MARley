"""Merged-pool retrieval strategy for the MARley pipeline.

Provides the MergedRetriever wrapper for multi-KB retrieval using the
merged-pool strategy: chunks from all knowledge bases are concatenated
into a single corpus and indexed by one inner retriever.
"""

from __future__ import annotations

from typing import Any

from src.marley.retrieval.base import RetrievalResult, Retriever


class MergedRetriever(Retriever):
    """Multi-KB retriever that merges all chunks into a single corpus.

    Wraps a single inner retriever (BM25, Vector, or Hybrid).  Multiple
    knowledge bases are concatenated before indexing, so the inner
    retriever searches across all KBs simultaneously.  This is the
    simplest multi-KB strategy and serves as the default.

    For cross-KB fusion with per-KB ranking, see FusionRetriever.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def index(self, corpus: list[dict[str, Any]]) -> None:
        """Index the merged corpus in the inner retriever."""
        self._retriever.index(corpus)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k results from the merged corpus."""
        return self._retriever.retrieve(query, k=k)

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return self._retriever.size
