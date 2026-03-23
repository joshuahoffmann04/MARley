"""Abstention evaluation runners.

Provides two evaluation modes:
- Level 1 sweep: Fast threshold sweep using only retrieval scores (no LLM).
- Full evaluation: Two-level abstention evaluation with generator.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from evaluation.utils import AbstentionMetrics, compute_abstention_metrics
from src.marley.abstention.detection import detect_abstention, extract_abstention_reason
from src.marley.models.generation import Generator
from src.marley.models.retrieval import Retriever
from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)


def run_level1_sweep(
    retriever: Retriever,
    corpus: list[dict],
    questions: list[dict],
    thresholds: list[float],
    *,
    k: int = 5,
    normalization_strategy: str = "vector",
    normalization_params: dict | None = None,
) -> list[dict]:
    """Sweep Level 1 thresholds without running the generator.

    For each threshold, determines which questions would trigger Level 1
    abstention based on retrieval confidence alone. This is fast because
    it only requires retrieval + scoring, not generation.

    Args:
        retriever: Retriever instance (must be indexed with corpus).
        corpus: Chunk corpus (used to index the retriever).
        questions: Evaluation questions, each with 'question_id', 'question',
            and 'expected_abstention'.
        thresholds: List of threshold values to sweep.
        k: Number of chunks to retrieve per query.
        normalization_strategy: Score normalization strategy.
        normalization_params: Extra kwargs for normalize_scores.

    Returns:
        List of dicts, one per threshold, each containing 'threshold'
        and 'metrics' (AbstentionMetrics as dict).
    """
    norm_params = normalization_params or {}

    # Index the retriever
    retriever.index(corpus)

    # Pre-compute normalized scores for all questions
    question_scores: list[dict] = []
    for q in questions:
        raw_results = retriever.retrieve(q["question"], k=k)
        normalized = normalize_scores(raw_results, normalization_strategy, **norm_params)
        confidence = compute_confidence(normalized)
        question_scores.append({
            "question_id": q["question_id"],
            "expected_abstention": q.get("expected_abstention", False),
            "confidence": confidence,
            "normalized_results": normalized,
        })

    # Sweep thresholds
    sweep_results: list[dict] = []
    for threshold in thresholds:
        eval_results = []
        for qs in question_scores:
            filtered = filter_by_threshold(qs["normalized_results"], threshold)
            system_abstained = len(filtered) == 0
            eval_results.append({
                "question_id": qs["question_id"],
                "expected_abstention": qs["expected_abstention"],
                "system_abstained": system_abstained,
                "abstention_level": 1 if system_abstained else None,
            })
        metrics = compute_abstention_metrics(eval_results, threshold)
        sweep_results.append({
            "threshold": threshold,
            "metrics": asdict(metrics),
        })

    return sweep_results


def run_abstention_evaluation(
    retriever: Retriever,
    generator: Generator,
    corpus: list[dict],
    questions: list[dict],
    *,
    k: int = 5,
    threshold: float = 0.3,
    normalization_strategy: str = "vector",
    normalization_params: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run full two-level abstention evaluation.

    For each question:
    1. Retrieve and normalize scores.
    2. Apply Level 1 threshold check.
    3. If passed: generate answer, check Level 2.
    4. Record outcome (abstained/answered, level, confidence).

    Args:
        retriever: Retriever instance (must be indexed with corpus).
        generator: Generator instance.
        corpus: Chunk corpus.
        questions: Evaluation questions with ground truth.
        k: Number of chunks to retrieve.
        threshold: Confidence threshold for Level 1.
        normalization_strategy: Score normalization strategy.
        normalization_params: Extra kwargs for normalize_scores.
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        Report dict with 'config', 'metrics', and 'results' keys.
    """
    norm_params = normalization_params or {}
    retriever.index(corpus)

    per_question_results: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions):
        raw_results = retriever.retrieve(q["question"], k=k)
        normalized = normalize_scores(raw_results, normalization_strategy, **norm_params)
        confidence = compute_confidence(normalized)
        filtered = filter_by_threshold(normalized, threshold)

        if not filtered:
            # Level 1 abstention
            per_question_results.append({
                "question_id": q["question_id"],
                "expected_abstention": q.get("expected_abstention", False),
                "system_abstained": True,
                "abstention_level": 1,
                "confidence": confidence,
                "answer": "",
                "reason": "retrieval confidence below threshold",
            })
        else:
            # Generate answer
            context = [
                {"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata}
                for r in filtered
            ]
            gen_result = generator.generate(q["question"], context)

            if detect_abstention(gen_result.answer):
                # Level 2 abstention
                per_question_results.append({
                    "question_id": q["question_id"],
                    "expected_abstention": q.get("expected_abstention", False),
                    "system_abstained": True,
                    "abstention_level": 2,
                    "confidence": confidence,
                    "answer": "",
                    "reason": extract_abstention_reason(gen_result.answer),
                })
            else:
                # Normal answer
                per_question_results.append({
                    "question_id": q["question_id"],
                    "expected_abstention": q.get("expected_abstention", False),
                    "system_abstained": False,
                    "abstention_level": None,
                    "confidence": confidence,
                    "answer": gen_result.answer,
                    "reason": "",
                })

        if progress_callback is not None:
            progress_callback(i + 1, total)

    metrics = compute_abstention_metrics(per_question_results, threshold)

    return {
        "config": {
            "k": k,
            "threshold": threshold,
            "normalization_strategy": normalization_strategy,
            "normalization_params": norm_params,
        },
        "metrics": asdict(metrics),
        "results": per_question_results,
    }


def run_and_report(
    retriever: Retriever,
    generator: Generator,
    eval_path: str | Path,
    corpus: list[dict],
    thresholds: list[float] | None = None,
    *,
    k: int = 5,
    knowledge_base: str = "",
    normalization_strategy: str = "vector",
    normalization_params: dict | None = None,
) -> dict:
    """Full pipeline: load data, sweep Level 1, run Level 2 at best threshold.

    Args:
        retriever: Retriever instance.
        generator: Generator instance.
        eval_path: Path to evaluation JSON file.
        corpus: Chunk corpus.
        thresholds: Thresholds to sweep (default: 0.0 to 1.0 in 0.05 steps).
        k: Number of chunks to retrieve.
        knowledge_base: Name of the knowledge base (for metadata).
        normalization_strategy: Score normalization strategy.
        normalization_params: Extra kwargs for normalize_scores.

    Returns:
        Comprehensive report dict with level1_sweep, level2_evaluation,
        and metadata.
    """
    if thresholds is None:
        thresholds = [round(i * 0.05, 2) for i in range(21)]

    # Load evaluation data
    path = Path(eval_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]

    # Level 1 sweep
    sweep = run_level1_sweep(
        retriever, corpus, questions, thresholds,
        k=k,
        normalization_strategy=normalization_strategy,
        normalization_params=normalization_params,
    )

    # Find best F1 threshold
    best = max(sweep, key=lambda s: s["metrics"]["f1"])
    best_threshold = best["threshold"]

    # Level 2 evaluation at best threshold
    evaluation = run_abstention_evaluation(
        retriever, generator, corpus, questions,
        k=k,
        threshold=best_threshold,
        normalization_strategy=normalization_strategy,
        normalization_params=normalization_params,
    )

    return {
        "knowledge_base": knowledge_base,
        "normalization_strategy": normalization_strategy,
        "level1_sweep": sweep,
        "best_threshold": best_threshold,
        "level2_evaluation": evaluation,
    }
