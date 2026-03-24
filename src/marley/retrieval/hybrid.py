"""Hybrid retrieval strategy using Reciprocal Rank Fusion (RRF).

Combines results from exactly two retriever instances (typically BM25
and Vector) by fusing their ranked lists using the weighted RRF formula:

    score(d) = sum( weight_i / (k_rrf + rank_i(d)) )

where rank_i(d) is the 1-based rank of document d in retriever i's
result list, weight_i defaults to 1.0 (uniform), and k_rrf is a
smoothing constant (default 60, from the original paper by Cormack,
Clarke & Buettcher, 2009).
"""

from __future__ import annotations

from src.marley.models.constants import DEFAULT_K_RRF_HYBRID
from src.marley.models.retrieval import rrf_fuse
from src.marley.retrieval.base import RetrievalResult, Retriever


class HybridRetriever(Retriever):
    """Within-KB hybrid retriever combining two retrievers via RRF.

    Fuses ranked lists from exactly two retrievers (typically one sparse
    and one dense) into a single ranking using Reciprocal Rank Fusion.
    Optional per-retriever weights allow tuning the balance between the
    two strategies.  RRF scores are typically in the range 0 to ~0.03,
    depending on k_rrf.
    """

    def __init__(
        self,
        retrievers: tuple[Retriever, Retriever],
        k_rrf: int = DEFAULT_K_RRF_HYBRID,
        weights: list[float] | None = None,
    ) -> None:
        if len(retrievers) != 2:
            msg = f"Expected exactly 2 retrievers, got {len(retrievers)}."
            raise ValueError(msg)
        if weights is not None and len(weights) != 2:
            msg = f"Expected 2 weights (one per retriever), got {len(weights)}."
            raise ValueError(msg)
        self._retrievers = retrievers
        self._k_rrf = k_rrf
        self._weights = weights

    def index(self, corpus: list[dict]) -> None:
        """Index the corpus in both sub-retrievers."""
        for retriever in self._retrievers:
            retriever.index(corpus)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k results by fusing ranked lists from both retrievers.

        Each sub-retriever is queried for k results.  The results are fused
        using RRF scores, and the top-k documents by fused score are returned.
        """
        all_results = [r.retrieve(query, k=k) for r in self._retrievers]
        return rrf_fuse(
            all_results, k_rrf=self._k_rrf, k=k, weights=self._weights,
        )

    @property
    def size(self) -> int:
        """Return the number of indexed chunks (from the first retriever)."""
        return self._retrievers[0].size
