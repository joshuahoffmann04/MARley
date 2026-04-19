"""Retrieval score normalization and threshold filtering.

Normalizes scores from different retriever types to a common [0, 1] range
so that a single confidence threshold can be applied regardless of the
underlying retrieval strategy.

Normalization strategies:
    bm25:   score / (score + k)  -- saturation function
    vector: identity             -- cosine similarity already in [0, 1]
    rrf:    score / max_score    -- divide by theoretical RRF maximum
"""

from __future__ import annotations

from src.marley.models.constants import DEFAULT_K_RRF, NORMALIZATION_STRATEGIES
from src.marley.models.retrieval import RetrievalResult


def normalize_scores(
    results: list[RetrievalResult],
    strategy: str,
    *,
    bm25_k: float = 1.0,
    rrf_n_retrievers: int = 2,
    rrf_k: int = DEFAULT_K_RRF,
) -> list[RetrievalResult]:
    """Normalize retrieval scores to [0, 1].

    Args:
        results: Retrieval results with raw scores.
        strategy: One of 'bm25', 'vector', 'rrf'.
        bm25_k: Saturation parameter for BM25 normalization.
            When raw score equals bm25_k, normalized value is 0.5.
        rrf_n_retrievers: Number of sub-retrievers used in RRF fusion.
        rrf_k: RRF smoothing constant (must match the value used
            during fusion).

    Returns:
        New RetrievalResult list with normalized scores in [0, 1].

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy not in NORMALIZATION_STRATEGIES:
        msg = (
            f"Unknown normalization strategy '{strategy}'. "
            f"Must be one of {sorted(NORMALIZATION_STRATEGIES)}"
        )
        raise ValueError(msg)

    if not results:
        return []

    if strategy == "bm25":
        return _normalize_bm25(results, bm25_k)
    if strategy == "vector":
        return _normalize_vector(results)
    return _normalize_rrf(results, rrf_n_retrievers, rrf_k)


def filter_by_threshold(
    results: list[RetrievalResult],
    threshold: float,
) -> list[RetrievalResult]:
    """Remove results with score below the threshold.

    Args:
        results: Retrieval results (should be normalized to [0, 1]).
        threshold: Minimum score to keep a result.

    Returns:
        Filtered list containing only results with score >= threshold.
    """
    return [r for r in results if r.score >= threshold]


def compute_confidence(results: list[RetrievalResult]) -> float:
    """Return the maximum score from the result set.

    This represents the system's confidence that at least one retrieved
    chunk is relevant to the query.  Returns 0.0 if the result set is
    empty.
    """
    if not results:
        return 0.0
    return max(r.score for r in results)


def compute_fusion_confidence(
    sub_retriever_results: list[list[RetrievalResult]],
    sub_normalization_strategy: str,
    *,
    bm25_k: float = 1.0,
) -> float:
    """Confidence for a FusionRetriever over disjoint knowledge bases.

    When each sub-retriever operates on a different corpus, every chunk
    appears in exactly one sub-retriever's ranking. The fused RRF score
    therefore carries no per-query variation (it equals
    ``1/(k_rrf+1)`` for every top-1 chunk and collapses to a constant
    after RRF normalisation). That defeats any confidence-based
    abstention decision on top of the fused output.

    The Phase-12 fix is to compute confidence *before* fusion: normalise
    each sub-retriever's raw scores with its native strategy and take
    the maximum confidence across sub-retrievers. A query on which at
    least one sub-retriever is confident yields a correspondingly high
    aggregate confidence; a query on which all sub-retrievers are weak
    yields a low one.

    Args:
        sub_retriever_results: One raw result list per sub-retriever,
            in the order returned by the fusion retriever's children.
        sub_normalization_strategy: ``"bm25"``, ``"vector"``, or
            ``"rrf"`` (for Hybrid sub-retrievers). All sub-retrievers
            share the same strategy because a single Fusion
            configuration uses a single retriever type.
        bm25_k: BM25 saturation parameter (only used when the strategy
            is ``"bm25"``).

    Returns:
        The maximum per-sub-retriever normalised top-1 score, in
        ``[0, 1]``. Returns ``0.0`` when every sub-retriever returned an
        empty result list.
    """
    if not sub_retriever_results:
        return 0.0

    confidences: list[float] = []
    for sub in sub_retriever_results:
        if not sub:
            continue
        normalised = normalize_scores(
            sub,
            sub_normalization_strategy,
            bm25_k=bm25_k,
        )
        confidences.append(compute_confidence(normalised))
    return max(confidences) if confidences else 0.0


# ---------------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------------


def _normalize_bm25(
    results: list[RetrievalResult],
    k: float,
) -> list[RetrievalResult]:
    """Apply saturation normalization: score / (score + k)."""
    return [
        RetrievalResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.score / (r.score + k) if r.score > 0 else 0.0,
            metadata=r.metadata,
        )
        for r in results
    ]


def _normalize_vector(
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Identity normalization -- cosine similarity is already in [0, 1]."""
    return [
        RetrievalResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.score,
            metadata=r.metadata,
        )
        for r in results
    ]


def _normalize_rrf(
    results: list[RetrievalResult],
    n_retrievers: int,
    k_rrf: int,
) -> list[RetrievalResult]:
    """Normalize RRF scores by dividing by theoretical maximum.

    The theoretical maximum RRF score for a document ranked #1 in all
    sub-retrievers is: n_retrievers / (k_rrf + 1).
    """
    max_score = n_retrievers / (k_rrf + 1)
    return [
        RetrievalResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.score / max_score if max_score > 0 else 0.0,
            metadata=r.metadata,
        )
        for r in results
    ]
