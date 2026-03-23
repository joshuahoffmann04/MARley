"""End-to-end evaluation metrics aggregation.

Computes per-configuration manual metrics and builds a comparison
matrix across all configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.manual.models import (
    EvaluationItem,
    Judgement,
    ManualJudgement,
)


@dataclass
class E2EConfigMetrics:
    """Aggregated metrics for a single E2E configuration."""

    config_name: str
    strict_accuracy: float                  # CORRECT / judged
    lenient_accuracy: float                 # (CORRECT + PARTIALLY_CORRECT) / judged
    strict_accuracy_by_category: dict[str, float] = field(default_factory=dict)
    lenient_accuracy_by_category: dict[str, float] = field(default_factory=dict)
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0
    abstention_f1: float = 0.0
    judgement_distribution: dict[str, int] = field(default_factory=dict)
    num_judged: int = 0
    num_total: int = 0


def compute_e2e_config_metrics(
    items: list[EvaluationItem],
    judgements: list[ManualJudgement],
    config_name: str = "",
) -> E2EConfigMetrics:
    """Compute E2E evaluation metrics for a single configuration.

    Extends the manual evaluation metrics with per-category accuracy
    breakdown (direct, multi_source) and abstention F1.

    Args:
        items: Evaluation items for this configuration.
        judgements: Deduplicated human judgements for these items.
        config_name: Configuration name for the report.

    Returns:
        E2EConfigMetrics with all computed values.
    """
    judgement_map = {j.item_id: j for j in judgements}
    item_map = {i.id: i for i in items}

    # Count judgement distribution
    distribution: dict[str, int] = {j.value: 0 for j in Judgement}
    for j in judgements:
        if j.item_id in item_map:
            distribution[j.judgement.value] = distribution.get(
                j.judgement.value, 0,
            ) + 1

    # Collect judged pairs
    judged_pairs: list[tuple[EvaluationItem, ManualJudgement]] = []
    for item in items:
        if item.id in judgement_map:
            judged_pairs.append((item, judgement_map[item.id]))

    num_judged = len(judged_pairs)
    num_total = len(items)

    if num_judged == 0:
        return E2EConfigMetrics(
            config_name=config_name,
            strict_accuracy=0.0,
            lenient_accuracy=0.0,
            num_judged=0,
            num_total=num_total,
            judgement_distribution=distribution,
        )

    # Overall accuracy
    strict_correct = sum(
        1 for _, j in judged_pairs if j.judgement == Judgement.CORRECT
    )
    lenient_correct = sum(
        1 for _, j in judged_pairs
        if j.judgement in {Judgement.CORRECT, Judgement.PARTIALLY_CORRECT}
    )

    # Include correct abstentions in accuracy
    # (correctly refusing is a "correct" system behavior)
    correct_abstention_count = sum(
        1 for _, j in judged_pairs
        if j.judgement == Judgement.CORRECT_ABSTENTION
    )

    strict_accuracy = (strict_correct + correct_abstention_count) / num_judged
    lenient_accuracy = (lenient_correct + correct_abstention_count) / num_judged

    # Per-category accuracy
    category_strict: dict[str, list[bool]] = {}
    category_lenient: dict[str, list[bool]] = {}

    for item, j in judged_pairs:
        cat = item.category
        if not cat:
            continue
        is_strict = j.judgement in {Judgement.CORRECT, Judgement.CORRECT_ABSTENTION}
        is_lenient = j.judgement in {
            Judgement.CORRECT,
            Judgement.PARTIALLY_CORRECT,
            Judgement.CORRECT_ABSTENTION,
        }
        category_strict.setdefault(cat, []).append(is_strict)
        category_lenient.setdefault(cat, []).append(is_lenient)

    strict_by_cat = {
        cat: sum(vals) / len(vals)
        for cat, vals in sorted(category_strict.items())
    }
    lenient_by_cat = {
        cat: sum(vals) / len(vals)
        for cat, vals in sorted(category_lenient.items())
    }

    # Abstention metrics
    correct_abs = distribution.get(Judgement.CORRECT_ABSTENTION.value, 0)
    incorrect_abs = distribution.get(Judgement.INCORRECT_ABSTENTION.value, 0)
    missing_abs = distribution.get(Judgement.MISSING_ABSTENTION.value, 0)

    total_abs = correct_abs + incorrect_abs
    abs_precision = correct_abs / total_abs if total_abs > 0 else 0.0

    total_unanswerable = correct_abs + missing_abs
    abs_recall = correct_abs / total_unanswerable if total_unanswerable > 0 else 0.0

    abs_f1 = (
        2 * abs_precision * abs_recall / (abs_precision + abs_recall)
        if (abs_precision + abs_recall) > 0
        else 0.0
    )

    return E2EConfigMetrics(
        config_name=config_name,
        strict_accuracy=strict_accuracy,
        lenient_accuracy=lenient_accuracy,
        strict_accuracy_by_category=strict_by_cat,
        lenient_accuracy_by_category=lenient_by_cat,
        abstention_precision=abs_precision,
        abstention_recall=abs_recall,
        abstention_f1=abs_f1,
        judgement_distribution=distribution,
        num_judged=num_judged,
        num_total=num_total,
    )


def build_comparison_table(
    config_metrics: list[E2EConfigMetrics],
) -> list[dict]:
    """Build a comparison table across all configurations.

    Returns a list of dicts (one per config) suitable for tabular
    display or DataFrame conversion, sorted by lenient accuracy
    descending.

    Each dict contains:
        config_name, strict_accuracy, lenient_accuracy,
        strict_direct, strict_multi_source, strict_unanswerable,
        abstention_precision, abstention_recall, abstention_f1,
        num_judged.
    """
    rows: list[dict] = []

    for m in config_metrics:
        rows.append({
            "config_name": m.config_name,
            "strict_accuracy": round(m.strict_accuracy, 4),
            "lenient_accuracy": round(m.lenient_accuracy, 4),
            "strict_direct": round(
                m.strict_accuracy_by_category.get("direct", 0.0), 4,
            ),
            "strict_multi_source": round(
                m.strict_accuracy_by_category.get("multi_source", 0.0), 4,
            ),
            "strict_unanswerable": round(
                m.strict_accuracy_by_category.get("unanswerable", 0.0), 4,
            ),
            "abstention_precision": round(m.abstention_precision, 4),
            "abstention_recall": round(m.abstention_recall, 4),
            "abstention_f1": round(m.abstention_f1, 4),
            "num_judged": m.num_judged,
        })

    rows.sort(key=lambda r: r["lenient_accuracy"], reverse=True)
    return rows
