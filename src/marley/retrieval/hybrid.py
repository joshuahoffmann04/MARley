"""Hybrid retrieval strategy using Reciprocal Rank Fusion (RRF).

Combines results from two retriever instances (typically BM25 and Vector)
by fusing their ranked lists using the RRF formula:

    score(d) = sum( 1 / (k_rrf + rank_i(d)) )

where rank_i(d) is the rank of document d in retriever i's result list,
and k_rrf is a smoothing constant (default 60, from the original paper).
"""

from __future__ import annotations

from src.marley.retrieval.base import RetrievalResult, Retriever

_DEFAULT_K_RRF = 60


class HybridRetriever(Retriever):
    """Hybrid retriever combining two retrievers via Reciprocal Rank Fusion.

    Args:
        retrievers: Tuple of exactly two Retriever instances.
        k_rrf: RRF smoothing constant (default 60).
    """

    def __init__(
        self,
        retrievers: tuple[Retriever, Retriever],
        k_rrf: int = _DEFAULT_K_RRF,
    ) -> None:
        if len(retrievers) != 2:
            msg = f"Expected exactly 2 retrievers, got {len(retrievers)}."
            raise ValueError(msg)
        self._retrievers = retrievers
        self._k_rrf = k_rrf

    def index(self, corpus: list[dict]) -> None:
        """Index the corpus in both sub-retrievers."""
        for retriever in self._retrievers:
            retriever.index(corpus)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k results by fusing ranked lists from both retrievers.

        Each sub-retriever is queried for k results. The results are fused
        using RRF scores, and the top-k documents by fused score are returned.
        """
        # Collect results from both retrievers
        all_results: list[list[RetrievalResult]] = []
        for retriever in self._retrievers:
            all_results.append(retriever.retrieve(query, k=k))

        # Build RRF scores
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}

        for results in all_results:
            for rank, result in enumerate(results):
                rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + (
                    1.0 / (self._k_rrf + rank + 1)
                )
                # Keep the result with the highest original score for metadata/text
                if result.chunk_id not in doc_map or result.score > doc_map[result.chunk_id].score:
                    doc_map[result.chunk_id] = result

        # Sort by RRF score descending and take top-k
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        return [
            RetrievalResult(
                chunk_id=cid,
                text=doc_map[cid].text,
                score=rrf_scores[cid],
                metadata=doc_map[cid].metadata,
            )
            for cid in sorted_ids
        ]

    @property
    def size(self) -> int:
        """Return the number of indexed documents (from the first retriever)."""
        return self._retrievers[0].size
