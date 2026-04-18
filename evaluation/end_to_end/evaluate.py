"""End-to-end evaluation runners for the MARley pipeline.

Runs the complete pipeline (retrieval -> abstention -> generation) for
each question in the evaluation dataset, and then scores every
non-abstained answerable answer with RAGAS. The result object carries
both the pipeline decisions and the answer-quality scores, so the
thesis-grade comparison table can rank configurations by correctness
rather than orchestration alone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.end_to_end.config import E2EConfig
from evaluation.judge import Judge
from evaluation.utils import compute_abstention_metrics, load_evaluation
from src.marley.abstention.detection import detect_abstention, extract_abstention_reason
from src.marley.models.generation import Generator
from src.marley.models.retrieval import Retriever
from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)

logger = logging.getLogger(__name__)


@dataclass
class E2EResult:
    """Result for a single question in a single configuration.

    ``faithfulness``, ``answer_relevance``, and ``correctness`` are
    populated only for samples that satisfy the Phase-11 scoring scope
    (``expected_abstention=False`` **and** ``abstained=False``). All
    other samples keep the ``NaN`` defaults so that downstream
    aggregations can exclude them cleanly.
    """

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
    faithfulness: float = float("nan")
    answer_relevance: float = float("nan")
    correctness: float = float("nan")


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
    judge: Judge,
    *,
    threshold: float,
    normalization_params: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[E2EResult]:
    """Run the end-to-end pipeline for a single configuration.

    Two passes:

    1. **Pipeline pass** — for each question, retrieve top-k, normalise
       scores, check Level-1 threshold, optionally generate, detect
       Level-2 abstention, and append an :class:`E2EResult`. Along the
       way, collect every sample that qualifies for answer-quality
       scoring (``expected_abstention=False`` **and** ``abstained=False``)
       into a side buffer carrying the result index plus the RAGAS
       inputs.
    2. **Scoring pass** — hand the buffer to :func:`_score_e2e_answers`,
       which delegates to RAGAS via the supplied judge. The returned
       per-metric lists are written back into the matching
       :class:`E2EResult` slots.

    Args:
        config: The E2E configuration.
        retriever: Already-indexed retriever (or FusionRetriever).
        generator: Generator instance (always Ollama).
        questions: Evaluation questions from load_questions().
        judge: Judge from :func:`evaluation.judge.make_judge` used to
            score eligible answers.
        threshold: Abstention threshold for Level 1.
        normalization_params: Extra kwargs for normalize_scores.
        progress_callback: Optional callback(current, total) invoked
            during the pipeline pass.

    Returns:
        List of E2EResult, one per question.
    """
    norm_params = normalization_params or {}
    total = len(questions)
    results: list[E2EResult] = []
    scoring_buffer: list[dict] = []

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
                # Normal answer — eligible for RAGAS scoring iff the
                # question was supposed to be answerable.
                result_index = len(results)
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
                if (
                    not q.get("expected_abstention", False)
                    and q.get("reference_answer", "")
                ):
                    scoring_buffer.append({
                        "index": result_index,
                        "question": q["question"],
                        "answer": gen_result.answer,
                        "reference": q["reference_answer"],
                        "context": context,
                    })

        if progress_callback is not None:
            progress_callback(i + 1, total)

    # --- Scoring pass ---------------------------------------------------
    if scoring_buffer:
        logger.info(
            "  Scoring %d answerable answers with RAGAS (batch_size=%d)...",
            len(scoring_buffer), judge.batch_size,
        )
        scores = _score_e2e_answers(scoring_buffer, judge)
        for buf, score in zip(scoring_buffer, scores, strict=True):
            r = results[buf["index"]]
            r.faithfulness = score["faithfulness"]
            r.answer_relevance = score["answer_relevancy"]
            r.correctness = score["factual_correctness"]

    return results


def _score_e2e_answers(buffer: list[dict], judge: Judge) -> list[dict]:
    """Batch-score answerable E2E answers with RAGAS.

    Delegates to the same RAGAS metric stack used by
    :func:`evaluation.generation.evaluate._score_with_ragas`. Each
    buffer entry is a dict with ``question``, ``answer``, ``reference``,
    and ``context`` (a list of chunk dicts with a ``text`` key).

    Returns a parallel list of dicts with ``faithfulness``,
    ``answer_relevancy``, and ``factual_correctness`` keys. Samples the
    judge cannot parse receive ``NaN`` (same fallback as the generation
    pipeline).
    """
    # Reuse the shared chunked retry helper so E2E inherits the same
    # failure semantics as the generation eval.
    from evaluation.generation.evaluate import _chunked_batch_score
    from ragas.metrics.collections import (
        AnswerRelevancy,
        Faithfulness,
        FactualCorrectness,
    )

    faithfulness_metric = Faithfulness(llm=judge.llm)
    relevancy_metric = AnswerRelevancy(llm=judge.llm, embeddings=judge.embeddings)
    correctness_metric = FactualCorrectness(llm=judge.llm)

    faithfulness_inputs = [
        {
            "user_input": b["question"],
            "response": b["answer"],
            "retrieved_contexts": [c.get("text", "") for c in b["context"]],
        }
        for b in buffer
    ]
    relevancy_inputs = [
        {"user_input": b["question"], "response": b["answer"]} for b in buffer
    ]
    correctness_inputs = [
        {"response": b["answer"], "reference": b["reference"]} for b in buffer
    ]

    logger.info("    Scoring faithfulness...")
    faith_results = _chunked_batch_score(
        faithfulness_metric, faithfulness_inputs, judge.batch_size,
    )
    logger.info("    Scoring answer relevancy...")
    relev_results = _chunked_batch_score(
        relevancy_metric, relevancy_inputs, judge.batch_size,
    )
    logger.info("    Scoring factual correctness...")
    correct_results = _chunked_batch_score(
        correctness_metric, correctness_inputs, judge.batch_size,
    )

    return [
        {
            "faithfulness": faith_results[i].value,
            "answer_relevancy": relev_results[i].value,
            "factual_correctness": correct_results[i].value,
        }
        for i in range(len(buffer))
    ]


def run_and_report(
    config: E2EConfig,
    retriever: Retriever,
    generator: Generator,
    questions: list[dict],
    judge: Judge,
    *,
    thresholds: list[float] | None = None,
    normalization_params: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Full pipeline: sweep threshold, run E2E evaluation, return report.

    Combines threshold optimisation, full pipeline execution, and
    RAGAS answer-quality scoring into a single convenience function.

    Args:
        config: The E2E configuration.
        retriever: Already-indexed retriever.
        generator: Generator instance (always Ollama).
        questions: Evaluation questions.
        judge: Judge for RAGAS scoring of eligible answers.
        thresholds: Threshold values for Level 1 sweep.
        normalization_params: Extra kwargs for normalize_scores.
        progress_callback: Optional callback(current, total) for generation.

    Returns:
        Report dict with ``config``, ``threshold``, ``level1_sweep``,
        ``abstention_metrics``, ``generation_metrics``,
        ``generator_model``, ``judge_batch_size``, and ``results``.
    """
    # Step 1: Sweep Level 1 thresholds (fast, no LLM)
    best_threshold, sweep = sweep_threshold(
        retriever, questions, config.normalization_strategy,
        k=config.k,
        thresholds=thresholds,
        normalization_params=normalization_params,
    )

    # Step 2: Run full pipeline at optimal threshold (incl. RAGAS scoring)
    results = run_e2e_config(
        config, retriever, generator, questions, judge,
        threshold=best_threshold,
        normalization_params=normalization_params,
        progress_callback=progress_callback,
    )

    # Step 3: Compute automatic abstention + generation metrics
    abstention_input = [
        {
            "expected_abstention": r.expected_abstention,
            "system_abstained": r.abstained,
        }
        for r in results
    ]
    abstention_metrics = compute_abstention_metrics(abstention_input, best_threshold)
    generation_metrics = _aggregate_generation_metrics(results)

    return {
        "config": asdict(config),
        "threshold": best_threshold,
        "level1_sweep": sweep,
        "abstention_metrics": asdict(abstention_metrics),
        "generation_metrics": generation_metrics,
        "generator_model": generator.model,
        "judge_batch_size": judge.batch_size,
        "results": [asdict(r) for r in results],
    }


def _aggregate_generation_metrics(results: list[E2EResult]) -> dict:
    """Aggregate per-result RAGAS scores into a report-friendly dict.

    NaN values are excluded from the three averages. ``num_scored``
    counts how many results contributed at least one non-NaN score —
    this equals the number of eligible samples run through the judge.
    """
    import math

    def _mean(values: list[float]) -> float:
        clean = [v for v in values if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    num_scored = sum(
        1 for r in results
        if not math.isnan(r.faithfulness)
        or not math.isnan(r.answer_relevance)
        or not math.isnan(r.correctness)
    )
    return {
        "num_scored": num_scored,
        "avg_faithfulness": _mean([r.faithfulness for r in results]),
        "avg_answer_relevance": _mean([r.answer_relevance for r in results]),
        "avg_correctness": _mean([r.correctness for r in results]),
    }


def save_report(report: dict, path: str | Path) -> None:
    """Save an E2E evaluation report to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
