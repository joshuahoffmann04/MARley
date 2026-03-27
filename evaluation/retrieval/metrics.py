"""Retrieval evaluation metrics for the MARley pipeline.

Implements Precision@k, Recall@k, MRR, MAP, F1@k, and Jaccard@k.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    map: float
    f1_at_k: float
    jaccard_at_k: float
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


def average_precision(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Average Precision at k for a single query.

    AP@k = (1 / |relevant|) * Σ_{i=1}^{k} (P@i * rel(i))

    where rel(i) = 1 if the document at rank i is relevant, 0 otherwise.
    Only the top-k positions are considered.

    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute F1@k for a single query.

    F1@k = 2 * P@k * R@k / (P@k + R@k)

    Returns 0.0 if both precision and recall are 0.
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def jaccard_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Jaccard similarity at k for a single query.

    Jaccard@k = |relevant ∩ retrieved[:k]| / |relevant ∪ retrieved[:k]|

    Returns 0.0 if both sets are empty.
    """
    top_k_set = set(retrieved[:k])
    intersection = len(relevant & top_k_set)
    union = len(relevant | top_k_set)
    if union == 0:
        return 0.0
    return intersection / union


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
            map=0.0,
            f1_at_k=0.0,
            jaccard_at_k=0.0,
            k=k,
            num_queries=0,
        )

    total_p = 0.0
    total_r = 0.0
    total_mrr = 0.0
    total_ap = 0.0
    total_f1 = 0.0
    total_jaccard = 0.0

    for retrieved, relevant in results:
        total_p += precision_at_k(retrieved, relevant, k)
        total_r += recall_at_k(retrieved, relevant, k)
        total_mrr += mrr(retrieved, relevant)
        total_ap += average_precision(retrieved, relevant, k)
        total_f1 += f1_at_k(retrieved, relevant, k)
        total_jaccard += jaccard_at_k(retrieved, relevant, k)

    n = len(results)
    return RetrievalMetrics(
        precision_at_k=total_p / n,
        recall_at_k=total_r / n,
        mrr=total_mrr / n,
        map=total_ap / n,
        f1_at_k=total_f1 / n,
        jaccard_at_k=total_jaccard / n,
        k=k,
        num_queries=n,
    )
