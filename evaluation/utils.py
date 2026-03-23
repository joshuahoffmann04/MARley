"""Shared utilities for evaluation modules.

Provides common functions used across retrieval, generation, abstention,
and end-to-end evaluation to avoid code duplication.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def load_json(path: str | Path) -> dict:
    """Load a JSON file and return its contents as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_evaluation(eval_path: str | Path) -> list[dict]:
    """Load an annotated evaluation JSON file.

    Returns the list of question dicts, each with 'id', 'question',
    'relevant_chunks', 'category', and 'expected_abstention'.
    """
    data = load_json(eval_path)
    return data["questions"]


def merge_chunks(*chunk_paths: str | Path) -> list[dict]:
    """Load and concatenate chunks from multiple JSON chunk files.

    Each file is expected to contain a ``"chunks"`` key with a list of
    chunk dicts.  Raises ``ValueError`` if duplicate ``chunk_id`` values
    are detected across files.
    """
    all_chunks: list[dict] = []
    seen_ids: set[str] = set()

    for path in chunk_paths:
        data = load_json(path)
        chunks = data["chunks"]
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid in seen_ids:
                msg = f"Duplicate chunk_id across files: {cid}"
                raise ValueError(msg)
            seen_ids.add(cid)
        all_chunks.extend(chunks)

    return all_chunks


def merge_evaluation_data(
    eval_paths: dict[str, str | Path],
) -> list[dict]:
    """Merge evaluation datasets from multiple KBs.

    Loads each evaluation file and merges ``relevant_chunks`` per
    question across all KBs using set union.  A question is included
    in the output only if it has at least one relevant chunk in any
    included KB or if it is marked as unanswerable
    (``expected_abstention``).

    Args:
        eval_paths: Mapping of KB name to evaluation file path.

    Returns:
        List of question dicts with merged ``relevant_chunks``.
    """
    questions_by_id: dict[str, dict] = {}

    for _kb_name, path in eval_paths.items():
        data = load_json(path)
        for q in data["questions"]:
            qid = q["id"]
            if qid not in questions_by_id:
                questions_by_id[qid] = {
                    "id": qid,
                    "question": q["question"],
                    "reference_answer": q.get("reference_answer", ""),
                    "category": q.get("category", ""),
                    "expected_abstention": q.get("expected_abstention", False),
                    "relevant_chunks": set(),
                }
            questions_by_id[qid]["relevant_chunks"].update(
                q.get("relevant_chunks", []),
            )

    result: list[dict] = []
    for q in questions_by_id.values():
        q["relevant_chunks"] = sorted(q["relevant_chunks"])
        result.append(q)

    return result


# ---------------------------------------------------------------------------
# Abstention metrics (shared between abstention and end-to-end evaluation)
# ---------------------------------------------------------------------------


@dataclass
class AbstentionMetrics:
    """Metrics for evaluating abstention behavior.

    Attributes:
        precision: correct_abstention / all_abstentions (1.0 if none).
        recall: correct_abstention / all_unanswerable (0.0 if none).
        f1: Harmonic mean of precision and recall.
        false_abstention_rate: incorrect_abstention / all_answerable.
        coverage: answered / total (proportion receiving an answer).
        num_correct_abstention: Unanswerable questions correctly abstained.
        num_incorrect_abstention: Answerable questions incorrectly abstained.
        num_missing_abstention: Unanswerable questions incorrectly answered.
        num_answered: Questions that received an answer.
        num_total: Total number of questions evaluated.
        threshold: The confidence threshold used for this evaluation.
    """

    precision: float
    recall: float
    f1: float
    false_abstention_rate: float
    coverage: float
    num_correct_abstention: int
    num_incorrect_abstention: int
    num_missing_abstention: int
    num_answered: int
    num_total: int
    threshold: float


def compute_abstention_metrics(
    results: list[dict],
    threshold: float,
) -> AbstentionMetrics:
    """Compute abstention metrics from evaluation results.

    Args:
        results: List of result dicts, each with:
            - expected_abstention (bool): ground truth
            - system_abstained (bool): system decision
        threshold: The confidence threshold (stored in metrics for reference).

    Returns:
        AbstentionMetrics with all computed values.
    """
    if not results:
        return AbstentionMetrics(
            precision=1.0, recall=1.0, f1=1.0,
            false_abstention_rate=0.0, coverage=1.0,
            num_correct_abstention=0, num_incorrect_abstention=0,
            num_missing_abstention=0, num_answered=0, num_total=0,
            threshold=threshold,
        )

    num_total = len(results)
    num_correct_abstention = 0
    num_incorrect_abstention = 0
    num_missing_abstention = 0
    num_answered = 0

    for r in results:
        expected = r["expected_abstention"]
        abstained = r["system_abstained"]

        if expected and abstained:
            num_correct_abstention += 1
        elif not expected and abstained:
            num_incorrect_abstention += 1
        elif expected and not abstained:
            num_missing_abstention += 1
        else:
            num_answered += 1

    total_abstentions = num_correct_abstention + num_incorrect_abstention
    precision = (
        num_correct_abstention / total_abstentions
        if total_abstentions > 0 else 1.0
    )

    total_unanswerable = num_correct_abstention + num_missing_abstention
    recall = (
        num_correct_abstention / total_unanswerable
        if total_unanswerable > 0 else 1.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    total_answerable = num_answered + num_incorrect_abstention
    false_abstention_rate = (
        num_incorrect_abstention / total_answerable
        if total_answerable > 0 else 0.0
    )

    total_answered = num_answered + num_missing_abstention
    coverage = total_answered / num_total

    return AbstentionMetrics(
        precision=precision, recall=recall, f1=f1,
        false_abstention_rate=false_abstention_rate, coverage=coverage,
        num_correct_abstention=num_correct_abstention,
        num_incorrect_abstention=num_incorrect_abstention,
        num_missing_abstention=num_missing_abstention,
        num_answered=num_answered, num_total=num_total,
        threshold=threshold,
    )
