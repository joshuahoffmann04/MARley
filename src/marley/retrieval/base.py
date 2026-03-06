"""Abstract base class for retrieval strategies.

All retrievers in the MARley pipeline implement the Retriever interface,
ensuring consistent behavior across BM25, vector, and hybrid approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """A single retrieval hit with its score."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class Retriever(ABC):
    """Abstract base for all retrieval strategies."""

    @abstractmethod
    def index(self, corpus: list[dict]) -> None:
        """Build the retrieval index from a list of chunk dicts.

        Each dict must have at least 'chunk_id', 'text', and 'metadata'.
        """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a query.

        Returns results sorted by descending relevance score.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed documents."""
