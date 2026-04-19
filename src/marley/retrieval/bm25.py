"""BM25 retrieval strategy for the MARley pipeline.

Uses the rank_bm25 library (Okapi BM25) with lowercase whitespace
tokenization as the baseline sparse retrieval method.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from src.marley.models.constants import DEFAULT_K
from src.marley.models.retrieval import load_chunks, validate_corpus
from src.marley.retrieval.base import RetrievalResult, Retriever

# Backward-compatible re-export so existing ``from bm25 import load_chunks``
# continues to work.  Canonical location: ``src.marley.models.retrieval``.
__all__ = ["BM25Retriever", "load_chunks"]


def _tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing and splitting on whitespace."""
    return text.lower().split()


class BM25Retriever(Retriever):
    """BM25 (Okapi) sparse retrieval over chunked documents.

    Uses lowercase whitespace tokenization and the rank_bm25 library
    with default parameters (k1=1.5, b=0.75).  Scores are unbounded
    (typically 0 to ~100+, higher means more relevant).  Results with
    a score of zero or below are filtered out.
    """

    def __init__(self) -> None:
        self._corpus: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def index(self, corpus: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of chunk dicts."""
        if not corpus:
            self._corpus = []
            self._bm25 = None
            return

        validate_corpus(corpus)
        self._corpus = corpus
        tokenized = [_tokenize(doc["text"]) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[RetrievalResult]:
        """Retrieve the top-k chunks most relevant to the query."""
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc = self._corpus[idx]
            results.append(RetrievalResult(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
            ))

        return results

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._corpus)
