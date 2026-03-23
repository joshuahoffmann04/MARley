"""RRF k-parameter tuning for HybridRetriever and FusionRetriever.

Provides sweep functions that evaluate retrieval quality across a range
of k_rrf values, enabling independent tuning of the RRF smoothing
constant for within-KB fusion (Hybrid) and cross-KB fusion (Fusion).

Reference:
    Cormack, Clarke & Buettcher (2009).  Reciprocal rank fusion
    outperforms condorcet and individual rank learning methods.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from evaluation.retrieval.metrics import RetrievalMetrics, evaluate_retriever
from evaluation.utils import load_evaluation, merge_chunks, merge_evaluation_data
from src.marley.models.constants import DEFAULT_K
from src.marley.models.retrieval import Retriever, rrf_fuse

# Default sweep values covering the practical range
DEFAULT_SWEEP_VALUES: list[int] = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]


# ---------------------------------------------------------------------------
# Hybrid k_rrf sweep
# ---------------------------------------------------------------------------


def sweep_hybrid_k_rrf(
    retriever_factory: Callable[[int], Retriever],
    chunk_path: str | Path,
    eval_path: str | Path,
    k: int = DEFAULT_K,
    sweep_values: list[int] | None = None,
    *,
    skip_unanswerable: bool = True,
) -> dict:
    """Sweep k_rrf for HybridRetriever on a single knowledge base.

    Creates a HybridRetriever(BM25, Vector) for each k_rrf value,
    indexes it on the provided chunks, and measures retrieval quality.

    Args:
        retriever_factory: Callable taking k_rrf and returning a fresh
            HybridRetriever instance (unindexed).
        chunk_path: Path to the chunk JSON file.
        eval_path: Path to the evaluation JSON file.
        k: Number of top results to consider.
        sweep_values: List of k_rrf values to test.

    Returns:
        Dict with ``best_k_rrf``, ``best_metrics``, and ``sweep_results``
        (list of ``{k_rrf, metrics}`` dicts).
    """
    if sweep_values is None:
        sweep_values = DEFAULT_SWEEP_VALUES

    data = json.loads(Path(chunk_path).read_text(encoding="utf-8"))
    corpus = data["chunks"]
    questions = load_evaluation(eval_path)

    sweep_results: list[dict] = []

    for k_rrf in sweep_values:
        retriever = retriever_factory(k_rrf)
        retriever.index(corpus)
        metrics = _evaluate_questions(retriever, questions, k, skip_unanswerable)
        sweep_results.append({
            "k_rrf": k_rrf,
            "metrics": asdict(metrics),
        })

    best = max(sweep_results, key=lambda r: r["metrics"]["recall_at_k"])

    return {
        "sweep_type": "hybrid",
        "best_k_rrf": best["k_rrf"],
        "best_metrics": best["metrics"],
        "sweep_results": sweep_results,
        "config": {
            "k": k,
            "sweep_values": sweep_values,
            "corpus_size": len(corpus),
            "num_questions": len(questions),
        },
    }


# ---------------------------------------------------------------------------
# Fusion k_rrf sweep
# ---------------------------------------------------------------------------


def sweep_fusion_k_rrf(
    retriever_factory: Callable[[], Retriever],
    chunk_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    k: int = DEFAULT_K,
    sweep_values: list[int] | None = None,
    *,
    skip_unanswerable: bool = True,
) -> dict:
    """Sweep k_rrf for FusionRetriever across multiple knowledge bases.

    Creates per-KB retrievers (one per KB, using the factory), indexes
    each on its own chunks, and tests FusionRetriever at each k_rrf value.

    To isolate the fusion-level RRF from the within-KB RRF, the factory
    should return a simple retriever (e.g., BM25Retriever), not a
    HybridRetriever.

    Args:
        retriever_factory: Callable returning a fresh Retriever instance.
        chunk_paths: KB name -> chunk file path mapping.
        eval_paths: KB name -> evaluation file path mapping.
        k: Number of top results to consider.
        sweep_values: List of k_rrf values to test.

    Returns:
        Dict with ``best_k_rrf``, ``best_metrics``, and ``sweep_results``.
    """
    if sweep_values is None:
        sweep_values = DEFAULT_SWEEP_VALUES

    # Build and index one retriever per KB (shared across all sweep values)
    per_kb_retrievers: dict[str, Retriever] = {}
    total_chunks = 0
    for kb_name, path in chunk_paths.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = data["chunks"]
        total_chunks += len(chunks)
        r = retriever_factory()
        r.index(chunks)
        per_kb_retrievers[kb_name] = r

    questions = merge_evaluation_data(eval_paths)

    sweep_results: list[dict] = []

    for k_rrf in sweep_values:
        pairs: list[tuple[list[str], set[str]]] = []
        for q in questions:
            if skip_unanswerable and q.get("expected_abstention", False):
                continue
            relevant = set(q.get("relevant_chunks", []))
            if not relevant:
                continue

            result_lists = [
                r.retrieve(q["question"], k=k) for r in per_kb_retrievers.values()
            ]
            fused = rrf_fuse(result_lists, k_rrf=k_rrf, k=k)
            retrieved_ids = [r.chunk_id for r in fused]
            pairs.append((retrieved_ids, relevant))

        metrics = evaluate_retriever(pairs, k=k)
        sweep_results.append({
            "k_rrf": k_rrf,
            "metrics": asdict(metrics),
        })

    best = max(sweep_results, key=lambda r: r["metrics"]["recall_at_k"])

    return {
        "sweep_type": "fusion",
        "best_k_rrf": best["k_rrf"],
        "best_metrics": best["metrics"],
        "sweep_results": sweep_results,
        "config": {
            "k": k,
            "sweep_values": sweep_values,
            "knowledge_bases": sorted(chunk_paths.keys()),
            "total_chunks": total_chunks,
            "num_questions": len(questions),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evaluate_questions(
    retriever: Retriever,
    questions: list[dict],
    k: int,
    skip_unanswerable: bool,
) -> RetrievalMetrics:
    """Run retrieval and compute metrics for a list of questions."""
    pairs: list[tuple[list[str], set[str]]] = []
    for q in questions:
        if skip_unanswerable and q.get("expected_abstention", False):
            continue
        relevant = set(q.get("relevant_chunks", []))
        if not relevant:
            continue
        results = retriever.retrieve(q["question"], k=k)
        retrieved_ids = [r.chunk_id for r in results]
        pairs.append((retrieved_ids, relevant))
    return evaluate_retriever(pairs, k=k)
