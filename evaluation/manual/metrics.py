"""Compute evaluation metrics from manual (human) judgements.

Joins evaluation items with their human judgements and computes
strict/lenient accuracy, abstention precision/recall, and
per-distractor-level breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.manual.models import (
    ABSTENTION_JUDGEMENTS,
    ANSWER_JUDGEMENTS,
    EvaluationItem,
    Judgement,
    ManualJudgement,
)


@dataclass
class ManualEvalMetrics:
    """Aggregated metrics from manual evaluation judgements."""

    strict_accuracy: float
    lenient_accuracy: float
    strict_accuracy_by_distractors: dict[int, float] = field(default_factory=dict)
    lenient_accuracy_by_distractors: dict[int, float] = field(default_factory=dict)
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0
    judgement_distribution: dict[str, int] = field(default_factory=dict)
    num_judged: int = 0
    num_total: int = 0
    knowledge_base: str = ""


def compute_manual_metrics(
    items: list[EvaluationItem],
    judgements: list[ManualJudgement],
    knowledge_base: str = "",
) -> ManualEvalMetrics:
    """Compute evaluation metrics from manual judgements.

    Joins items with their latest judgement (``load_judgements``
    already deduplicates). Items without a judgement are excluded
    from metric computation.

    Accuracy is computed in two variants:

    - **Strict:** only ``correct`` counts as correct.
    - **Lenient:** ``correct`` + ``partially_correct`` count as correct.

    Both are reported overall and grouped by distractor level.

    Abstention metrics:

    - **Precision:** ``correct_abstention / (correct_abstention + incorrect_abstention)``
    - **Recall:** ``correct_abstention / (correct_abstention + missing_abstention)``

    Args:
        items: All evaluation items (for total count and metadata).
        judgements: Deduplicated list of human judgements.
        knowledge_base: Knowledge base identifier for the report.

    Returns:
        ManualEvalMetrics with all computed values.
    """
    judgement_map = {j.item_id: j for j in judgements}
    item_map = {i.id: i for i in items}

    # Count judgement distribution
    distribution: dict[str, int] = {j.value: 0 for j in Judgement}
    for j in judgements:
        if j.item_id in item_map:
            distribution[j.judgement.value] = distribution.get(j.judgement.value, 0) + 1

    # Collect judged items with their judgements
    judged_pairs: list[tuple[EvaluationItem, ManualJudgement]] = []
    for item in items:
        if item.id in judgement_map:
            judged_pairs.append((item, judgement_map[item.id]))

    num_judged = len(judged_pairs)
    num_total = len(items)

    if num_judged == 0:
        return ManualEvalMetrics(
            strict_accuracy=0.0,
            lenient_accuracy=0.0,
            abstention_precision=0.0,
            abstention_recall=0.0,
            judgement_distribution=distribution,
            num_judged=0,
            num_total=num_total,
            knowledge_base=knowledge_base,
        )

    # Compute overall accuracy
    strict_correct = sum(
        1 for _, j in judged_pairs if j.judgement == Judgement.CORRECT
    )
    lenient_correct = sum(
        1 for _, j in judged_pairs
        if j.judgement in {Judgement.CORRECT, Judgement.PARTIALLY_CORRECT}
    )

    strict_accuracy = strict_correct / num_judged
    lenient_accuracy = lenient_correct / num_judged

    # Compute accuracy by distractor level
    by_level_strict: dict[int, list[bool]] = {}
    by_level_lenient: dict[int, list[bool]] = {}

    for item, j in judged_pairs:
        n_dist = item.metadata.get("num_distractors")
        if n_dist is None:
            continue
        by_level_strict.setdefault(n_dist, []).append(
            j.judgement == Judgement.CORRECT
        )
        by_level_lenient.setdefault(n_dist, []).append(
            j.judgement in {Judgement.CORRECT, Judgement.PARTIALLY_CORRECT}
        )

    strict_by_dist = {
        level: sum(vals) / len(vals)
        for level, vals in sorted(by_level_strict.items())
    }
    lenient_by_dist = {
        level: sum(vals) / len(vals)
        for level, vals in sorted(by_level_lenient.items())
    }

    # Abstention metrics
    correct_abstention_count = distribution.get(Judgement.CORRECT_ABSTENTION.value, 0)
    incorrect_abstention_count = distribution.get(Judgement.INCORRECT_ABSTENTION.value, 0)
    missing_abstention_count = distribution.get(Judgement.MISSING_ABSTENTION.value, 0)

    total_abstentions = correct_abstention_count + incorrect_abstention_count
    abstention_precision = (
        correct_abstention_count / total_abstentions if total_abstentions > 0 else 0.0
    )

    total_unanswerable = correct_abstention_count + missing_abstention_count
    abstention_recall = (
        correct_abstention_count / total_unanswerable if total_unanswerable > 0 else 0.0
    )

    return ManualEvalMetrics(
        strict_accuracy=strict_accuracy,
        lenient_accuracy=lenient_accuracy,
        strict_accuracy_by_distractors=strict_by_dist,
        lenient_accuracy_by_distractors=lenient_by_dist,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        judgement_distribution=distribution,
        num_judged=num_judged,
        num_total=num_total,
        knowledge_base=knowledge_base,
    )
