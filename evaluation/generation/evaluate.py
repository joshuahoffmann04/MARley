"""Evaluation runner for the MARley generation pipeline.

For each answerable question, the runner assembles context from the
ground-truth relevant chunks plus a variable number of BM25-ranked
distractor chunks and generates an answer.

Quality is measured via RAGAS:
  - Faithfulness (answer grounded in retrieved context)
  - Answer Relevancy (answer addresses the question)
  - Factual Correctness (answer matches the reference answer)

Distractor selection is deterministic: distractors are non-relevant
chunks ranked by BM25 similarity to the question, simulating realistic
retrieval noise.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from pathlib import Path

from evaluation.generation.metrics import (
    GenerationEvalResult,
    compute_generation_metrics,
)
from evaluation.judge import Judge
from evaluation.utils import load_evaluation
from src.marley.models.generation import Generator
from src.marley.retrieval import BM25Retriever

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

# RAGAS metric key names (used in score dicts returned by _score_with_ragas)
_RAGAS_KEY_FAITHFULNESS = "faithfulness"
_RAGAS_KEY_ANSWER_RELEVANCY = "answer_relevancy"
_RAGAS_KEY_FACTUAL_CORRECTNESS = "factual_correctness"

_SAMPLE_MAX_RETRIES = 3
"""Maximum retries for a single failed RAGAS sample before using NaN."""

# --- Distractor selection ----------------------------------------------------


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

    id_to_chunk = {c["chunk_id"]: c for c in non_relevant}
    return [id_to_chunk[r.chunk_id] for r in results if r.chunk_id in id_to_chunk]


def _assemble_context(
    relevant_chunks: list[dict],
    distractors: list[dict],
    num_distractors: int,
    seed: int,
) -> list[dict]:
    """Combine relevant chunks with N distractors and shuffle.

    Uses a fixed seed per question to ensure deterministic but
    unpredictable chunk ordering for the LLM.
    """
    selected = relevant_chunks + distractors[:num_distractors]
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


# --- RAGAS scoring -----------------------------------------------------------


def _chunked_batch_score(metric, inputs: list[dict], batch_size: int):
    """Score inputs in small chunks sized to the active judge backend.

    If a batch fails (e.g. LLM producing invalid structured output),
    retries each failed sample individually up to 3 times before
    falling back to NaN.

    Args:
        metric: A RAGAS Collections metric with batch_score().
        inputs: List of input dicts for the metric.
        batch_size: Number of samples per batch_score() call. Comes from
            the active :class:`~evaluation.judge.Judge` (Ollama: 20, OpenAI: 50).

    Returns:
        List of MetricResult objects (one per input).
    """
    from ragas.metrics.result import MetricResult

    results: list[MetricResult] = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i:i + batch_size]
        try:
            results.extend(metric.batch_score(chunk))
        except Exception:
            logger.warning("      Batch %d–%d failed, retrying per-sample",
                           i, min(i + batch_size, len(inputs)))
            for j, sample in enumerate(chunk):
                for attempt in range(1, _SAMPLE_MAX_RETRIES + 1):
                    try:
                        results.append(metric.score(**sample))
                        break
                    except Exception:
                        if attempt < _SAMPLE_MAX_RETRIES:
                            logger.warning("      Sample %d attempt %d/%d failed, retrying",
                                           i + j, attempt, _SAMPLE_MAX_RETRIES)
                        else:
                            logger.warning("      Sample %d failed after %d attempts, using NaN",
                                           i + j, _SAMPLE_MAX_RETRIES)
                            results.append(MetricResult(value=float("nan")))
        logger.info("      %d / %d scored", min(i + batch_size, len(inputs)), len(inputs))
    return results


def _score_with_ragas(
    raw_results: list[dict],
    judge: Judge,
) -> list[dict]:
    """Score generation results using RAGAS metrics.

    Uses RAGAS 0.4.x Collections API with chunked batch_score() calls.
    The chunk size comes from ``judge.batch_size``, which is tuned per
    backend so the Ollama path stays sequential-friendly and the OpenAI
    path exploits its async concurrency.

    Args:
        raw_results: List of dicts with 'question', 'generated_answer',
            'reference_answer', and 'context' keys.
        judge: The active judge (LLM + embeddings + batch size) from
            :func:`evaluation.judge.make_judge`.

    Returns:
        List of score dicts with 'faithfulness', 'answer_relevancy',
        and 'factual_correctness' keys (one per result).
    """
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
            "user_input": r["question"],
            "response": r["generated_answer"],
            "retrieved_contexts": [c.get("text", "") for c in r["context"]],
        }
        for r in raw_results
    ]

    relevancy_inputs = [
        {
            "user_input": r["question"],
            "response": r["generated_answer"],
        }
        for r in raw_results
    ]

    correctness_inputs = [
        {
            "response": r["generated_answer"],
            "reference": r["reference_answer"],
        }
        for r in raw_results
    ]

    logger.info(
        "  Running RAGAS evaluation (%d samples, batch_size=%d)...",
        len(raw_results),
        judge.batch_size,
    )

    logger.info("    Scoring faithfulness...")
    faith_results = _chunked_batch_score(
        faithfulness_metric, faithfulness_inputs, judge.batch_size
    )
    logger.info("    Scoring answer relevancy...")
    relev_results = _chunked_batch_score(
        relevancy_metric, relevancy_inputs, judge.batch_size
    )
    logger.info("    Scoring factual correctness...")
    correct_results = _chunked_batch_score(
        correctness_metric, correctness_inputs, judge.batch_size
    )

    scores = []
    for i in range(len(raw_results)):
        scores.append({
            "faithfulness": faith_results[i].value,
            "answer_relevancy": relev_results[i].value,
            "factual_correctness": correct_results[i].value,
        })

    return scores


# --- Public API --------------------------------------------------------------


def run_generation_evaluation(
    generator: Generator,
    corpus: list[dict],
    questions: list[dict],
    judge: Judge,
    distractor_levels: list[int] | None = None,
    *,
    progress_callback=None,
) -> list[GenerationEvalResult]:
    """Run generation evaluation over all questions and distractor levels.

    Generates answers for each (question, distractor_level) pair and
    scores them using RAGAS (Faithfulness, Answer Relevancy, Correctness)
    via the supplied ``judge``.

    Args:
        generator: The generator to evaluate (always Ollama-backed).
        corpus: Full chunk corpus for distractor selection.
        questions: Annotated question dicts from an evaluation file.
        judge: Judge object from :func:`evaluation.judge.make_judge` that
            scores the generated answers.
        distractor_levels: List of distractor counts to test (default 0-10).
        progress_callback: Optional callable(question_id, num_distractors)
            invoked before each generation call.

    Returns:
        List of GenerationEvalResult for every question x level combination.
    """
    if distractor_levels is None:
        distractor_levels = list(range(11))

    corpus_map = {c["chunk_id"]: c for c in corpus}
    raw_results: list[dict] = []

    for q in questions:
        if q.get("expected_abstention", False):
            continue
        relevant_ids = set(q.get("relevant_chunks", []))
        if not relevant_ids:
            continue

        relevant_chunks = [
            corpus_map[cid] for cid in relevant_ids if cid in corpus_map
        ]
        if not relevant_chunks:
            continue

        max_needed = max(distractor_levels) if distractor_levels else 0
        distractors = select_distractors(
            q["question"], relevant_ids, corpus, max_distractors=max_needed,
        )
        seed = hash(q["id"]) & 0xFFFFFFFF

        for n_dist in distractor_levels:
            if progress_callback:
                progress_callback(q["id"], n_dist)

            context = _assemble_context(
                relevant_chunks, distractors, n_dist, seed + n_dist,
            )
            gen_result = generator.generate(q["question"], context)

            raw_results.append({
                "question_id": q["id"],
                "question": q["question"],
                "num_distractors": n_dist,
                "generated_answer": gen_result.answer,
                "reference_answer": q["reference_answer"],
                "context": context,
                "context_chunk_ids": gen_result.context_chunk_ids,
            })

    if not raw_results:
        return []

    # --- Score with RAGAS ---
    ragas_scores = _score_with_ragas(raw_results, judge)

    # --- Assemble final results ---
    results: list[GenerationEvalResult] = []
    for i, r in enumerate(raw_results):
        scores = ragas_scores[i]
        results.append(GenerationEvalResult(
            question_id=r["question_id"],
            num_distractors=r["num_distractors"],
            generated_answer=r["generated_answer"],
            reference_answer=r["reference_answer"],
            context_chunk_ids=r["context_chunk_ids"],
            faithfulness=scores.get(_RAGAS_KEY_FAITHFULNESS, 0.0),
            answer_relevance=scores.get(_RAGAS_KEY_ANSWER_RELEVANCY, 0.0),
            correctness=scores.get(_RAGAS_KEY_FACTUAL_CORRECTNESS, 0.0),
        ))

    return results


def run_and_report(
    generator: Generator,
    corpus: list[dict],
    eval_path: str | Path,
    judge: Judge,
    distractor_levels: list[int] | None = None,
    *,
    knowledge_base: str = "",
    subset: int | None = None,
    progress_callback=None,
) -> dict:
    """Load evaluation data, run generation evaluation, and return a report.

    Args:
        generator: The generator under evaluation (always Ollama).
        corpus: Chunk corpus used as both ground truth and distractor pool.
        eval_path: Path to the JSON eval file for this KB.
        judge: Judge object from :func:`evaluation.judge.make_judge`.
        distractor_levels: Distractor counts to test (default 0-10).
        knowledge_base: KB label stored in the output report.
        subset: If set, use only the first N questions — intended for
            quick subset-verification runs, not production reports.
        progress_callback: Optional progress callback.

    Returns:
        Dict with ``eval_file``, ``config``, ``metrics``, and ``results``.
    """
    questions = load_evaluation(eval_path)
    if subset is not None:
        questions = questions[:subset]

    eval_results = run_generation_evaluation(
        generator, corpus, questions,
        judge,
        distractor_levels=distractor_levels,
        progress_callback=progress_callback,
    )

    metrics = compute_generation_metrics(
        eval_results,
        knowledge_base=knowledge_base,
        model=generator.model,
    )

    config: dict = {
        "distractor_levels": distractor_levels or list(range(11)),
        "generator_model": generator.model,
        "judge_batch_size": judge.batch_size,
        "corpus_size": len(corpus),
        "knowledge_base": knowledge_base,
    }
    if subset is not None:
        config["subset"] = subset

    return {
        "eval_file": str(eval_path),
        "config": config,
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in eval_results],
    }
