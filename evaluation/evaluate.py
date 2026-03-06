"""Evaluation runner for the MARley retrieval pipeline.

Loads an annotated evaluation dataset and a set of chunks, runs a
retriever against all queries, and computes Precision@k, Recall@k,
and MRR.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from evaluation.metrics import RetrievalMetrics, evaluate_retriever
from src.marley.retrieval.base import Retriever


def load_evaluation(eval_path: str | Path) -> list[dict]:
    """Load an annotated evaluation JSON file.

    Returns the list of question dicts, each with 'id', 'question',
    'relevant_chunks', 'category', and 'expected_abstention'.
    """
    path = Path(eval_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def run_evaluation(
    retriever: Retriever,
    questions: list[dict],
    k: int = 5,
    *,
    skip_unanswerable: bool = True,
) -> RetrievalMetrics:
    """Run retrieval evaluation over a set of annotated questions.

    Args:
        retriever: An indexed Retriever instance.
        questions: List of question dicts from an evaluation file.
        k: Number of top results to consider.
        skip_unanswerable: If True, skip questions with expected_abstention=True
            (they have no relevant chunks by definition).

    Returns:
        Aggregated RetrievalMetrics.
    """
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


def run_and_report(
    retriever: Retriever,
    eval_path: str | Path,
    k: int = 5,
    *,
    skip_unanswerable: bool = True,
) -> dict:
    """Load evaluation data, run evaluation, and return a report dict.

    Returns a dict with 'eval_file', 'metrics', and 'config'.
    """
    questions = load_evaluation(eval_path)
    metrics = run_evaluation(
        retriever, questions, k=k,
        skip_unanswerable=skip_unanswerable,
    )

    return {
        "eval_file": str(eval_path),
        "config": {
            "k": k,
            "skip_unanswerable": skip_unanswerable,
            "retriever_type": type(retriever).__name__,
            "corpus_size": retriever.size,
        },
        "metrics": asdict(metrics),
    }
