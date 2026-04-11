"""Cross-KB retriever using Reciprocal Rank Fusion (RRF).

Provides the ``FusionRetriever`` wrapper for cross-KB retrieval.
The shared ``rrf_fuse`` algorithm lives in ``src.marley.models.retrieval``
and is re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from src.marley.models.constants import DEFAULT_K_RRF_FUSION
from src.marley.models.retrieval import rrf_fuse
from src.marley.retrieval.base import RetrievalResult, Retriever

# Re-export for backward compatibility
__all__ = ["FusionRetriever", "rrf_fuse"]


class FusionRetriever(Retriever):
    """Cross-KB retriever that fuses pre-indexed retrievers via RRF.

    Each sub-retriever must be independently indexed before constructing
    this wrapper.  Calling ``retrieve()`` runs all sub-retrievers and
    fuses their ranked lists via Reciprocal Rank Fusion.

    Designed for combined-KB configurations where each knowledge base has
    its own dedicated retriever.  Optional per-retriever weights allow
    boosting or dampening individual knowledge bases.  RRF scores are
    typically in the range 0 to ~0.03, depending on k_rrf and the number
    of sub-retrievers.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        k_rrf: int = DEFAULT_K_RRF_FUSION,
        weights: list[float] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("FusionRetriever requires at least one sub-retriever.")
        if weights is not None and len(weights) != len(retrievers):
            msg = (
                f"Expected {len(retrievers)} weights "
                f"(one per retriever), got {len(weights)}."
            )
            raise ValueError(msg)
        self._retrievers = list(retrievers)
        self._k_rrf = k_rrf
        self._weights = weights

    def index(self, corpus: list[dict[str, Any]]) -> None:
        """Not supported — index each sub-retriever independently."""
        raise NotImplementedError(
            "FusionRetriever wraps pre-indexed retrievers. "
            "Index the individual retrievers before constructing FusionRetriever."
        )

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k results by fusing ranked lists from all sub-retrievers."""
        result_lists = [r.retrieve(query, k=k) for r in self._retrievers]
        return rrf_fuse(
            result_lists, k_rrf=self._k_rrf, k=k, weights=self._weights,
        )

    @property
    def size(self) -> int:
        """Return the total number of indexed chunks across all sub-retrievers."""
        return sum(r.size for r in self._retrievers)
