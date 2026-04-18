"""Combined knowledge base generation evaluation for the MARley pipeline.

Extends the single-KB generation evaluation to combined-KB configurations.
For each question, ground-truth relevant chunks are drawn from multiple
KBs (set union), and distractors are selected from the merged corpus.

This uses the same controlled methodology as the single-KB evaluation:
ground-truth relevant chunks + BM25-ranked distractors at variable levels.
The only difference is the source of relevant chunks and the distractor
pool, enabling direct comparison with single-KB baselines.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from evaluation.generation.evaluate import run_generation_evaluation
from evaluation.generation.metrics import (
    GenerationEvalResult,
    compute_generation_metrics,
)
from evaluation.judge import Judge
from evaluation.utils import merge_chunks, merge_evaluation_data
from src.marley.models.generation import Generator


def run_combined_generation_evaluation(
    generator: Generator,
    chunk_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    judge: Judge,
    distractor_levels: list[int] | None = None,
    *,
    progress_callback=None,
) -> list[GenerationEvalResult]:
    """Run generation evaluation with merged multi-KB context.

    1. Merges chunks from all specified KBs into a single corpus.
    2. Merges evaluation data (``relevant_chunks`` via set union).
    3. Delegates to the standard generation evaluation runner.

    The merged corpus serves both as the source of ground-truth
    relevant chunks and as the distractor pool.  BM25-ranked
    distractor selection operates over the full merged corpus,
    providing harder distractors from a larger pool.

    Args:
        generator: The generator to evaluate.
        chunk_paths: KB name -> chunk file path mapping.
        eval_paths: KB name -> evaluation file path mapping.
        judge: Judge object from :func:`evaluation.judge.make_judge`.
        distractor_levels: Distractor counts to test (default 0-10).
        progress_callback: Optional ``callable(question_id, num_distractors)``
            invoked before each generation call.

    Returns:
        List of GenerationEvalResult for every question x level.
    """
    merged_corpus = merge_chunks(*chunk_paths.values())
    questions = merge_evaluation_data(eval_paths)

    return run_generation_evaluation(
        generator,
        merged_corpus,
        questions,
        judge,
        distractor_levels=distractor_levels,
        progress_callback=progress_callback,
    )


def run_and_report_combined(
    generator: Generator,
    chunk_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    judge: Judge,
    distractor_levels: list[int] | None = None,
    *,
    combination_name: str = "",
    progress_callback=None,
) -> dict:
    """Merge data, run combined generation evaluation, and return a report.

    Convenience wrapper that loads and merges multi-KB data, runs the
    generation evaluation, computes aggregated metrics, and returns a
    standardised report dict.

    Args:
        generator: The generator to evaluate.
        chunk_paths: KB name -> chunk file path mapping.
        eval_paths: KB name -> evaluation file path mapping.
        judge: Judge object from :func:`evaluation.judge.make_judge`.
        distractor_levels: Distractor counts to test (default 0-10).
        combination_name: Human-readable combination label
            (default: KB names joined with ``+``).
        progress_callback: Optional ``callable(question_id, num_distractors)``
            invoked before each generation call.

    Returns:
        Report dict with ``combination``, ``eval_files``, ``config``,
        ``metrics``, and ``results`` keys.
    """
    merged_corpus = merge_chunks(*chunk_paths.values())
    questions = merge_evaluation_data(eval_paths)

    eval_results = run_generation_evaluation(
        generator,
        merged_corpus,
        questions,
        judge,
        distractor_levels=distractor_levels,
        progress_callback=progress_callback,
    )

    kb_names = sorted(chunk_paths.keys())
    combination = combination_name or "+".join(kb_names)

    metrics = compute_generation_metrics(
        eval_results,
        knowledge_base=combination,
        model=generator.model,
    )

    return {
        "combination": combination,
        "eval_files": {kb: str(p) for kb, p in eval_paths.items()},
        "config": {
            "distractor_levels": distractor_levels or list(range(11)),
            "generator_model": generator.model,
            "judge_batch_size": judge.batch_size,
            "corpus_size": len(merged_corpus),
            "knowledge_bases": kb_names,
            "combination": combination,
        },
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in eval_results],
    }
