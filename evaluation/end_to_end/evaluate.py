"""End-to-end evaluation runners for the MARley pipeline.

Runs the complete pipeline (retrieval -> abstention -> generation) for
each question in the evaluation dataset, producing raw results that
can be converted to manual evaluation items.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.end_to_end.config import E2EConfig
from evaluation.utils import compute_abstention_metrics, load_evaluation
from src.marley.abstention.detection import detect_abstention, extract_abstention_reason
from src.marley.models.generation import Generator
from src.marley.models.retrieval import Retriever
from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)


@dataclass
class E2EResult:
    """Result for a single question in a single configuration."""

    question_id: str
    question: str
    reference_answer: str
    category: str
    expected_abstention: bool
    answer: str
    abstained: bool
    abstention_level: int | None
    abstention_reason: str
    confidence: float
    retrieval_chunk_ids: list[str]
    model: str


# Backward-compatible alias
load_questions = load_evaluation


def sweep_threshold(
    retriever: Retriever,
    questions: list[dict],
    normalization_strategy: str,
    *,
    k: int = 5,
    thresholds: list[float] | None = None,
    normalization_params: dict | None = None,
) -> tuple[float, list[dict]]:
    """Sweep Level 1 thresholds and return the optimal threshold.

    Runs retrieval for all questions once, then tests each threshold
    to determine which questions would trigger Level 1 abstention.
    Selects the threshold that maximizes F1.

    No LLM calls required -- this is a retrieval-only operation.

    Args:
        retriever: Already-indexed retriever.
        questions: Evaluation questions (must have 'id', 'question',
            'expected_abstention').
        normalization_strategy: Score normalization strategy.
        k: Number of chunks to retrieve per query.
        thresholds: Threshold values to sweep (default: 0.0 to 1.0
            in 0.05 steps).
        normalization_params: Extra kwargs for normalize_scores.

    Returns:
        Tuple of (best_threshold, sweep_results) where sweep_results
        is a list of {threshold, metrics} dicts.
    """
    if thresholds is None:
        thresholds = [round(i * 0.05, 2) for i in range(21)]
    norm_params = normalization_params or {}

    # Pre-compute normalized scores for all questions
    question_scores: list[dict] = []
    for q in questions:
        raw_results = retriever.retrieve(q["question"], k=k)
        normalized = normalize_scores(
            raw_results, normalization_strategy, **norm_params,
        )
        confidence = compute_confidence(normalized)
        question_scores.append({
            "question_id": q["id"],
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
                "expected_abstention": qs["expected_abstention"],
                "system_abstained": system_abstained,
            })
        metrics = compute_abstention_metrics(eval_results, threshold)
        sweep_results.append({
            "threshold": threshold,
            "metrics": asdict(metrics),
        })

    # Find best F1
    best = max(sweep_results, key=lambda s: s["metrics"]["f1"])
    return best["threshold"], sweep_results


def run_e2e_config(
    config: E2EConfig,
    retriever: Retriever,
    generator: Generator,
    questions: list[dict],
    *,
    threshold: float,
    normalization_params: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[E2EResult]:
    """Run the end-to-end pipeline for a single configuration.

    For each question:
    1. Retrieve top-k chunks from the (already-indexed) retriever.
    2. Normalize scores, compute confidence.
    3. Apply Level 1 threshold check.
    4. If passed: generate answer, check Level 2.
    5. Record E2EResult.

    Args:
        config: The E2E configuration.
        retriever: Already-indexed retriever (or FusionRetriever).
        generator: Generator instance.
        questions: Evaluation questions from load_questions().
        threshold: Abstention threshold for Level 1.
        normalization_params: Extra kwargs for normalize_scores.
        progress_callback: Optional callback(current, total).

    Returns:
        List of E2EResult, one per question.
    """
    norm_params = normalization_params or {}
    total = len(questions)
    results: list[E2EResult] = []

    for i, q in enumerate(questions):
        raw_results = retriever.retrieve(q["question"], k=config.k)
        normalized = normalize_scores(
            raw_results, config.normalization_strategy, **norm_params,
        )
        confidence = compute_confidence(normalized)
        filtered = filter_by_threshold(normalized, threshold)

        if not filtered:
            # Level 1 abstention
            results.append(E2EResult(
                question_id=q["id"],
                question=q["question"],
                reference_answer=q.get("reference_answer", ""),
                category=q.get("category", ""),
                expected_abstention=q.get("expected_abstention", False),
                answer="",
                abstained=True,
                abstention_level=1,
                abstention_reason="retrieval confidence below threshold",
                confidence=confidence,
                retrieval_chunk_ids=[r.chunk_id for r in normalized],
                model="",
            ))
        else:
            # Generate answer
            context = [
                {"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata}
                for r in filtered
            ]
            gen_result = generator.generate(q["question"], context)

            if detect_abstention(gen_result.answer):
                # Level 2 abstention
                results.append(E2EResult(
                    question_id=q["id"],
                    question=q["question"],
                    reference_answer=q.get("reference_answer", ""),
                    category=q.get("category", ""),
                    expected_abstention=q.get("expected_abstention", False),
                    answer="",
                    abstained=True,
                    abstention_level=2,
                    abstention_reason=extract_abstention_reason(gen_result.answer),
                    confidence=confidence,
                    retrieval_chunk_ids=[r.chunk_id for r in filtered],
                    model=gen_result.model,
                ))
            else:
                # Normal answer
                results.append(E2EResult(
                    question_id=q["id"],
                    question=q["question"],
                    reference_answer=q.get("reference_answer", ""),
                    category=q.get("category", ""),
                    expected_abstention=q.get("expected_abstention", False),
                    answer=gen_result.answer,
                    abstained=False,
                    abstention_level=None,
                    abstention_reason="",
                    confidence=confidence,
                    retrieval_chunk_ids=[r.chunk_id for r in filtered],
                    model=gen_result.model,
                ))

        if progress_callback is not None:
            progress_callback(i + 1, total)

    return results


def run_and_report(
    config: E2EConfig,
    retriever: Retriever,
    generator: Generator,
    questions: list[dict],
    *,
    thresholds: list[float] | None = None,
    normalization_params: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Full pipeline: sweep threshold, run E2E evaluation, return report.

    Combines threshold optimization and full pipeline execution into
    a single convenience function.

    Args:
        config: The E2E configuration.
        retriever: Already-indexed retriever.
        generator: Generator instance.
        questions: Evaluation questions.
        thresholds: Threshold values for Level 1 sweep.
        normalization_params: Extra kwargs for normalize_scores.
        progress_callback: Optional callback(current, total) for generation.

    Returns:
        Report dict with config, threshold, level1_sweep, abstention_metrics,
        generator_model, and results.
    """
    # Step 1: Sweep Level 1 thresholds (fast, no LLM)
    best_threshold, sweep = sweep_threshold(
        retriever, questions, config.normalization_strategy,
        k=config.k,
        thresholds=thresholds,
        normalization_params=normalization_params,
    )

    # Step 2: Run full pipeline at optimal threshold
    results = run_e2e_config(
        config, retriever, generator, questions,
        threshold=best_threshold,
        normalization_params=normalization_params,
        progress_callback=progress_callback,
    )

    # Step 3: Compute automatic abstention metrics
    abstention_input = [
        {
            "expected_abstention": r.expected_abstention,
            "system_abstained": r.abstained,
        }
        for r in results
    ]
    abstention_metrics = compute_abstention_metrics(abstention_input, best_threshold)

    return {
        "config": asdict(config),
        "threshold": best_threshold,
        "level1_sweep": sweep,
        "abstention_metrics": asdict(abstention_metrics),
        "generator_model": generator.model,
        "results": [asdict(r) for r in results],
    }


def save_report(report: dict, path: str | Path) -> None:
    """Save an E2E evaluation report to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
