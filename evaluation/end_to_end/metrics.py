"""End-to-end evaluation metrics aggregation.

Computes automatic metrics from ``E2EResult`` lists and builds a
comparison matrix across all configurations. Two metric families are
aggregated side by side:

* **Abstention metrics** — precision / recall / F1 derived from the
  orchestration decisions (answer vs. abstain), plus per-category
  breakdowns. Deterministic, no LLM involved.
* **Generation metrics** — RAGAS Faithfulness, Answer Relevancy,
  Factual Correctness averaged over non-abstained answerable samples
  (Option-A scoring scope, see ``run_e2e_config``). NaN values are
  excluded from the averages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evaluation.end_to_end.evaluate import E2EResult


@dataclass
class E2EConfigMetrics:
    """Aggregated automatic metrics for a single E2E configuration."""

    config_name: str
    num_total: int
    num_abstained: int
    abstention_rate: float
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    abstention_by_category: dict[str, dict] = field(default_factory=dict)
    avg_confidence: float = 0.0
    level1_abstentions: int = 0
    level2_abstentions: int = 0
    num_scored: int = 0
    avg_faithfulness: float = float("nan")
    avg_answer_relevance: float = float("nan")
    avg_correctness: float = float("nan")


def _nanmean(values: list[float]) -> float:
    """Mean over values, excluding NaN. Returns NaN if the list is empty."""
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def compute_e2e_config_metrics(
    results: list[E2EResult],
    config_name: str = "",
) -> E2EConfigMetrics:
    """Compute automatic E2E metrics from pipeline results.

    Computes abstention precision, recall, F1, and per-category
    breakdown from the orchestration decisions; then aggregates the
    RAGAS scores attached to non-abstained answerable samples.

    Abstention categories:
    - Correct abstention: expected_abstention=True AND abstained=True
    - Incorrect abstention: expected_abstention=False AND abstained=True
    - Missing abstention: expected_abstention=True AND abstained=False

    Args:
        results: List of E2EResult from run_e2e_config().
        config_name: Configuration name for the report.

    Returns:
        E2EConfigMetrics with all computed values.
    """
    num_total = len(results)
    if num_total == 0:
        return E2EConfigMetrics(
            config_name=config_name,
            num_total=0,
            num_abstained=0,
            abstention_rate=0.0,
            abstention_precision=0.0,
            abstention_recall=0.0,
            abstention_f1=0.0,
        )

    num_abstained = sum(1 for r in results if r.abstained)
    abstention_rate = num_abstained / num_total
    level1 = sum(1 for r in results if r.abstained and r.abstention_level == 1)
    level2 = sum(1 for r in results if r.abstained and r.abstention_level == 2)
    avg_confidence = sum(r.confidence for r in results) / num_total

    # Overall abstention precision / recall / F1
    correct_abs = sum(1 for r in results if r.expected_abstention and r.abstained)
    incorrect_abs = sum(1 for r in results if not r.expected_abstention and r.abstained)
    missing_abs = sum(1 for r in results if r.expected_abstention and not r.abstained)

    total_predicted = correct_abs + incorrect_abs
    precision = correct_abs / total_predicted if total_predicted > 0 else 0.0

    total_unanswerable = correct_abs + missing_abs
    recall = correct_abs / total_unanswerable if total_unanswerable > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Per-category breakdown
    categories = sorted({r.category for r in results})
    by_category: dict[str, dict] = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_correct = sum(1 for r in cat_results if r.expected_abstention and r.abstained)
        cat_incorrect = sum(1 for r in cat_results if not r.expected_abstention and r.abstained)
        cat_missing = sum(1 for r in cat_results if r.expected_abstention and not r.abstained)
        cat_predicted = cat_correct + cat_incorrect
        cat_prec = cat_correct / cat_predicted if cat_predicted > 0 else 0.0
        cat_unanswerable = cat_correct + cat_missing
        cat_rec = cat_correct / cat_unanswerable if cat_unanswerable > 0 else 0.0
        cat_f1 = (
            2 * cat_prec * cat_rec / (cat_prec + cat_rec)
            if (cat_prec + cat_rec) > 0
            else 0.0
        )
        by_category[cat] = {
            "num_total": len(cat_results),
            "num_abstained": sum(1 for r in cat_results if r.abstained),
            "abstention_precision": round(cat_prec, 4),
            "abstention_recall": round(cat_rec, 4),
            "abstention_f1": round(cat_f1, 4),
        }

    # RAGAS aggregates over the Option-A scored subset.
    num_scored = sum(
        1 for r in results
        if not math.isnan(r.faithfulness)
        or not math.isnan(r.answer_relevance)
        or not math.isnan(r.correctness)
    )
    avg_faith = _nanmean([r.faithfulness for r in results])
    avg_relev = _nanmean([r.answer_relevance for r in results])
    avg_correct = _nanmean([r.correctness for r in results])

    return E2EConfigMetrics(
        config_name=config_name,
        num_total=num_total,
        num_abstained=num_abstained,
        abstention_rate=abstention_rate,
        abstention_precision=precision,
        abstention_recall=recall,
        abstention_f1=f1,
        abstention_by_category=by_category,
        avg_confidence=avg_confidence,
        level1_abstentions=level1,
        level2_abstentions=level2,
        num_scored=num_scored,
        avg_faithfulness=avg_faith,
        avg_answer_relevance=avg_relev,
        avg_correctness=avg_correct,
    )


def _round_or_nan(value: float, digits: int = 4) -> float:
    """Round a float to ``digits`` decimals; pass NaN through untouched."""
    return value if math.isnan(value) else round(value, digits)


def build_comparison_table(
    config_metrics: list[E2EConfigMetrics],
) -> list[dict]:
    """Build a comparison table across all configurations.

    Returns a list of dicts (one per config) suitable for tabular
    display or DataFrame conversion, sorted by abstention_f1 descending.

    Each dict contains abstention columns (config_name, num_total,
    num_abstained, abstention_rate, abstention_precision,
    abstention_recall, abstention_f1, avg_confidence, level1_abstentions,
    level2_abstentions) **and** generation columns (num_scored,
    avg_faithfulness, avg_answer_relevance, avg_correctness).
    """
    rows: list[dict] = []
    for m in config_metrics:
        rows.append({
            "config_name": m.config_name,
            "num_total": m.num_total,
            "num_abstained": m.num_abstained,
            "abstention_rate": round(m.abstention_rate, 4),
            "abstention_precision": round(m.abstention_precision, 4),
            "abstention_recall": round(m.abstention_recall, 4),
            "abstention_f1": round(m.abstention_f1, 4),
            "avg_confidence": round(m.avg_confidence, 4),
            "level1_abstentions": m.level1_abstentions,
            "level2_abstentions": m.level2_abstentions,
            "num_scored": m.num_scored,
            "avg_faithfulness": _round_or_nan(m.avg_faithfulness),
            "avg_answer_relevance": _round_or_nan(m.avg_answer_relevance),
            "avg_correctness": _round_or_nan(m.avg_correctness),
        })
    rows.sort(key=lambda r: r["abstention_f1"], reverse=True)
    return rows
