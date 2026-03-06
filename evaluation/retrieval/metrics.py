"""Retrieval evaluation metrics for the MARley pipeline.

Implements Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)
as specified in the thesis proposal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    k: int
    num_queries: int


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@k for a single query.

    Precision@k = |relevant ∩ retrieved[:k]| / k

    Returns 0.0 if k is 0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@k for a single query.

    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Compute Reciprocal Rank for a single query.

    RR = 1 / rank of the first relevant document in retrieved.

    Returns 0.0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retriever(
    results: list[tuple[list[str], set[str]]],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute aggregated metrics over multiple queries.

    Args:
        results: List of (retrieved_ids, relevant_ids) tuples.
            Each retrieved_ids is ordered by descending relevance.
            Each relevant_ids is the ground-truth set.
        k: Number of top results to consider.

    Returns:
        Aggregated RetrievalMetrics averaged over all queries.
    """
    if not results:
        return RetrievalMetrics(
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            k=k,
            num_queries=0,
        )

    total_p = 0.0
    total_r = 0.0
    total_mrr = 0.0

    for retrieved, relevant in results:
        total_p += precision_at_k(retrieved, relevant, k)
        total_r += recall_at_k(retrieved, relevant, k)
        total_mrr += mrr(retrieved, relevant)

    n = len(results)
    return RetrievalMetrics(
        precision_at_k=total_p / n,
        recall_at_k=total_r / n,
        mrr=total_mrr / n,
        k=k,
        num_queries=n,
    )
