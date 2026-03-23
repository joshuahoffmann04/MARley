"""Evaluation runner for the MARley generation pipeline.

For each answerable question, the runner assembles context from the
ground-truth relevant chunks plus a variable number of BM25-ranked
distractor chunks and generates an answer.

Correctness assessment is performed separately via the manual
evaluation framework (``evaluation.manual``).

Distractor selection is deterministic: distractors are non-relevant
chunks ranked by BM25 similarity to the question, simulating
realistic retrieval noise.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

from evaluation.generation.metrics import (
    GenerationEvalResult,
    compute_generation_metrics,
)
from evaluation.utils import load_evaluation
from src.marley.models.generation import Generator
from src.marley.retrieval import BM25Retriever


def select_distractors(
    question: str,
    relevant_ids: set[str],
    corpus: list[dict],
    max_distractors: int = 10,
) -> list[dict]:
    """Select deterministic distractors ranked by BM25 similarity.

    Indexes only non-relevant chunks into a BM25 retriever, then
    retrieves the top-max_distractors by query similarity. This
    produces the hardest (most confusing) distractors first.

    Args:
        question: The query text.
        relevant_ids: Set of chunk IDs that are ground-truth relevant.
        corpus: Full chunk corpus.
        max_distractors: Maximum number of distractors to return.

    Returns:
        List of non-relevant chunk dicts, ranked by BM25 similarity.
    """
    non_relevant = [c for c in corpus if c["chunk_id"] not in relevant_ids]
    if not non_relevant or max_distractors <= 0:
        return []

    retriever = BM25Retriever()
    retriever.index(non_relevant)
    results = retriever.retrieve(question, k=max_distractors)

    # Map back to full chunk dicts
    id_to_chunk = {c["chunk_id"]: c for c in non_relevant}
    return [id_to_chunk[r.chunk_id] for r in results if r.chunk_id in id_to_chunk]


def _assemble_context(
    relevant_chunks: list[dict],
    distractors: list[dict],
    num_distractors: int,
    seed: int,
) -> list[dict]:
    """Combine relevant chunks with N distractors and shuffle.

    Uses a fixed seed per question to ensure the chunk order is
    deterministic but not predictable by the LLM.
    """
    selected = relevant_chunks + distractors[:num_distractors]
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def run_generation_evaluation(
    generator: Generator,
    corpus: list[dict],
    questions: list[dict],
    distractor_levels: list[int] | None = None,
    *,
    progress_callback=None,
) -> list[GenerationEvalResult]:
    """Run generation evaluation over all questions and distractor levels.

    Generates answers for each (question, distractor_level) pair.
    Correctness assessment is handled separately by the manual
    evaluation framework.

    Args:
        generator: The generator to evaluate.
        corpus: Full chunk corpus for distractor selection.
        questions: Annotated question dicts from an evaluation file.
        distractor_levels: List of distractor counts to test (default 0-10).
        progress_callback: Optional callable(question_id, num_distractors)
            invoked before each generation call.

    Returns:
        List of GenerationEvalResult for every question x level combination.
    """
    if distractor_levels is None:
        distractor_levels = list(range(11))

    corpus_map = {c["chunk_id"]: c for c in corpus}
    results: list[GenerationEvalResult] = []

    for q in questions:
        # Skip unanswerable and questions without ground truth
        if q.get("expected_abstention", False):
            continue
        relevant_ids = set(q.get("relevant_chunks", []))
        if not relevant_ids:
            continue

        # Resolve relevant chunk dicts
        relevant_chunks = [
            corpus_map[cid] for cid in relevant_ids if cid in corpus_map
        ]
        if not relevant_chunks:
            continue

        # Pre-compute distractors (BM25-ranked, deterministic)
        max_needed = max(distractor_levels) if distractor_levels else 0
        distractors = select_distractors(
            q["question"], relevant_ids, corpus, max_distractors=max_needed,
        )

        # Deterministic seed from question ID
        seed = hash(q["id"]) & 0xFFFFFFFF

        for n_dist in distractor_levels:
            if progress_callback:
                progress_callback(q["id"], n_dist)

            context = _assemble_context(relevant_chunks, distractors, n_dist, seed + n_dist)

            gen_result = generator.generate(q["question"], context)

            results.append(GenerationEvalResult(
                question_id=q["id"],
                num_distractors=n_dist,
                generated_answer=gen_result.answer,
                reference_answer=q["reference_answer"],
                context_chunk_ids=gen_result.context_chunk_ids,
            ))

    return results


def run_and_report(
    generator: Generator,
    corpus: list[dict],
    eval_path: str | Path,
    distractor_levels: list[int] | None = None,
    *,
    knowledge_base: str = "",
    progress_callback=None,
) -> dict:
    """Load evaluation data, run generation evaluation, and return a report.

    Returns a dict with 'eval_file', 'config', 'metrics', and 'results'.
    """
    questions = load_evaluation(eval_path)
    eval_results = run_generation_evaluation(
        generator, corpus, questions,
        distractor_levels=distractor_levels,
        progress_callback=progress_callback,
    )

    metrics = compute_generation_metrics(
        eval_results,
        knowledge_base=knowledge_base,
        model=generator.model,
    )

    return {
        "eval_file": str(eval_path),
        "config": {
            "distractor_levels": distractor_levels or list(range(11)),
            "generator_model": generator.model,
            "corpus_size": len(corpus),
            "knowledge_base": knowledge_base,
        },
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in eval_results],
    }
