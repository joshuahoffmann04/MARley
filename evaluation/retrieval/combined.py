"""Combined knowledge base retrieval evaluation for the MARley pipeline.

Provides two strategies for evaluating retrieval across multiple
knowledge bases:

1. **Merged Pool** — all chunks are combined into a single corpus and
   indexed in one retriever.
2. **Separate Retrieval + Fusion** — each KB gets its own retriever
   instance; per-query results are fused via Reciprocal Rank Fusion.

Both strategies reuse the existing retrieval metrics (Precision@k,
Recall@k, MRR, MAP, F1@k, Jaccard@k) and evaluation infrastructure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from evaluation.retrieval.metrics import RetrievalMetrics, evaluate_retriever
from evaluation.utils import merge_chunks, merge_evaluation_data
from src.marley.models.constants import DEFAULT_K, DEFAULT_K_RRF
from src.marley.models.retrieval import RetrievalResult, Retriever, rrf_fuse


# ---------------------------------------------------------------------------
# Strategy 1: Merged Pool
# ---------------------------------------------------------------------------


def run_merged_pool_evaluation(
    retriever: Retriever,
    chunk_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    k: int = DEFAULT_K,
    *,
    skip_unanswerable: bool = True,
) -> dict:
    """Evaluate retrieval with all chunks merged into a single corpus.

    1. Loads and merges chunks from all specified KBs.
    2. Indexes the retriever on the merged corpus.
    3. Loads and merges evaluation data (relevant_chunks across KBs).
    4. Evaluates with standard metrics.

    Returns:
        Report dict with ``strategy``, ``combination``, ``config``,
        and ``metrics`` keys.
    """
    merged_corpus = merge_chunks(*chunk_paths.values())
    retriever.index(merged_corpus)

    questions = merge_evaluation_data(eval_paths)
    metrics = _evaluate_questions(retriever, questions, k, skip_unanswerable)

    return _build_report(
        strategy="merged_pool",
        kb_names=sorted(chunk_paths.keys()),
        retriever=retriever,
        corpus_size=len(merged_corpus),
        k=k,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Strategy 2: Separate Retrieval + Fusion
# ---------------------------------------------------------------------------


def run_fusion_evaluation(
    retriever_factory: Callable[[], Retriever],
    chunk_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    k: int = DEFAULT_K,
    k_rrf: int = DEFAULT_K_RRF,
    *,
    skip_unanswerable: bool = True,
) -> dict:
    """Evaluate retrieval with per-KB retrievers fused via RRF.

    1. Creates one retriever per KB and indexes each on its own chunks.
    2. Loads and merges evaluation data.
    3. For each query: retrieves from all KB retrievers, fuses via RRF.
    4. Evaluates fused results with standard metrics.

    Args:
        retriever_factory: Callable returning a fresh Retriever instance.
        chunk_paths: KB name → chunk file path mapping.
        eval_paths: KB name → evaluation file path mapping.
        k: Number of top results to consider.
        k_rrf: RRF smoothing constant.
    """
    # Build one retriever per KB
    retrievers: dict[str, Retriever] = {}
    total_chunks = 0
    for kb_name, path in chunk_paths.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = data["chunks"]
        total_chunks += len(chunks)
        r = retriever_factory()
        r.index(chunks)
        retrievers[kb_name] = r

    questions = merge_evaluation_data(eval_paths)

    # Evaluate with per-query fusion
    pairs: list[tuple[list[str], set[str]]] = []
    for q in questions:
        if skip_unanswerable and q.get("expected_abstention", False):
            continue
        relevant = set(q.get("relevant_chunks", []))
        if not relevant:
            continue

        result_lists = [
            r.retrieve(q["question"], k=k) for r in retrievers.values()
        ]
        fused = rrf_fuse(result_lists, k_rrf=k_rrf, k=k)
        retrieved_ids = [r.chunk_id for r in fused]
        pairs.append((retrieved_ids, relevant))

    metrics = evaluate_retriever(pairs, k=k)

    return _build_report(
        strategy="fusion",
        kb_names=sorted(chunk_paths.keys()),
        retriever_type=type(retriever_factory()).__name__,
        corpus_size=total_chunks,
        k=k,
        metrics=metrics,
        k_rrf=k_rrf,
    )


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


def _build_report(
    *,
    strategy: str,
    kb_names: list[str],
    k: int,
    metrics: RetrievalMetrics,
    corpus_size: int,
    retriever: Retriever | None = None,
    retriever_type: str | None = None,
    k_rrf: int | None = None,
) -> dict:
    """Build a standardized report dict."""
    report: dict = {
        "strategy": strategy,
        "combination": "+".join(kb_names),
        "config": {
            "k": k,
            "retriever_type": retriever_type or type(retriever).__name__,
            "corpus_size": corpus_size,
            "knowledge_bases": kb_names,
        },
        "metrics": asdict(metrics),
    }
    if k_rrf is not None:
        report["config"]["k_rrf"] = k_rrf
    return report
