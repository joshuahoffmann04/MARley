"""BM25 retrieval strategy for the MARley pipeline.

Uses the rank_bm25 library (Okapi BM25) with lowercase whitespace
tokenization as the baseline sparse retrieval method.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.marley.retrieval.base import RetrievalResult, Retriever


def _tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing and splitting on whitespace."""
    return text.lower().split()


def load_chunks(chunk_path: str | Path) -> list[dict]:
    """Load chunks from a JSON file produced by the chunking pipeline.

    Supports both StPO chunk format and FAQ chunk format.
    Returns a list of dicts with 'chunk_id', 'text', and 'metadata'.
    """
    path = Path(chunk_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["chunks"]


class BM25Retriever(Retriever):
    """BM25 (Okapi) sparse retrieval over chunked documents."""

    def __init__(self) -> None:
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def index(self, corpus: list[dict]) -> None:
        """Build the BM25 index from a list of chunk dicts.

        Each dict must have at least 'chunk_id' and 'text'.
        """
        if not corpus:
            self._corpus = []
            self._bm25 = None
            return

        self._corpus = corpus
        tokenized = [_tokenize(doc["text"]) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k chunks most relevant to the query."""
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_query = _tokenize(query)
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
        """Return the number of indexed documents."""
        return len(self._corpus)
