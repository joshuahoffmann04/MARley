"""Tests for manual evaluation metrics computation.

Covers compute_manual_metrics() with various judgement distributions,
distractor level breakdowns, and abstention metrics.
"""

from __future__ import annotations

import pytest

from evaluation.manual.metrics import ManualEvalMetrics, compute_manual_metrics
from evaluation.manual.models import EvaluationItem, Judgement, ManualJudgement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(item_id: str, n_dist: int = 0, expected_abstention: bool = False) -> EvaluationItem:
    """Create a minimal EvaluationItem for testing."""
    return EvaluationItem(
        id=item_id,
        question="Q?",
        generated_answer="A.",
        reference_answer="R.",
        category="direct" if not expected_abstention else "unanswerable",
        expected_abstention=expected_abstention,
        metadata={"num_distractors": n_dist, "knowledge_base": "stpo"},
    )


def _judge(item_id: str, judgement: Judgement) -> ManualJudgement:
    """Create a ManualJudgement for testing."""
    return ManualJudgement(
        item_id=item_id,
        judgement=judgement,
        timestamp="2026-03-10T12:00:00",
    )


# ---------------------------------------------------------------------------
# TestComputeManualMetrics
# ---------------------------------------------------------------------------


class TestComputeManualMetrics:
    """Tests for compute_manual_metrics()."""

    def test_empty_judgements(self):
        items = [_item("i1"), _item("i2")]
        m = compute_manual_metrics(items, [], knowledge_base="stpo")
        assert m.strict_accuracy == 0.0
        assert m.lenient_accuracy == 0.0
        assert m.num_judged == 0
        assert m.num_total == 2

    def test_all_correct(self):
        items = [_item("i1"), _item("i2")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.CORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        assert m.strict_accuracy == 1.0
        assert m.lenient_accuracy == 1.0
        assert m.num_judged == 2

    def test_mixed_judgements_strict_vs_lenient(self):
        items = [_item("i1"), _item("i2"), _item("i3"), _item("i4")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.PARTIALLY_CORRECT),
            _judge("i3", Judgement.INCORRECT),
            _judge("i4", Judgement.CORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        assert m.strict_accuracy == 0.5       # 2/4
        assert m.lenient_accuracy == 0.75      # 3/4

    def test_all_incorrect(self):
        items = [_item("i1"), _item("i2")]
        judgements = [
            _judge("i1", Judgement.INCORRECT),
            _judge("i2", Judgement.INCORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        assert m.strict_accuracy == 0.0
        assert m.lenient_accuracy == 0.0

    def test_accuracy_by_distractor_level(self):
        items = [
            _item("i1", n_dist=0),
            _item("i2", n_dist=0),
            _item("i3", n_dist=5),
            _item("i4", n_dist=5),
        ]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.CORRECT),
            _judge("i3", Judgement.CORRECT),
            _judge("i4", Judgement.INCORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        assert m.strict_accuracy_by_distractors[0] == 1.0
        assert m.strict_accuracy_by_distractors[5] == 0.5

    def test_distractor_levels_sorted(self):
        items = [_item("i1", n_dist=10), _item("i2", n_dist=0), _item("i3", n_dist=5)]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.CORRECT),
            _judge("i3", Judgement.CORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        levels = list(m.strict_accuracy_by_distractors.keys())
        assert levels == [0, 5, 10]

    def test_abstention_precision(self):
        items = [
            _item("i1", expected_abstention=True),
            _item("i2"),
        ]
        judgements = [
            _judge("i1", Judgement.CORRECT_ABSTENTION),
            _judge("i2", Judgement.INCORRECT_ABSTENTION),
        ]
        m = compute_manual_metrics(items, judgements)
        # precision = 1 / (1 + 1) = 0.5
        assert m.abstention_precision == 0.5

    def test_abstention_recall(self):
        items = [
            _item("i1", expected_abstention=True),
            _item("i2", expected_abstention=True),
        ]
        judgements = [
            _judge("i1", Judgement.CORRECT_ABSTENTION),
            _judge("i2", Judgement.MISSING_ABSTENTION),
        ]
        m = compute_manual_metrics(items, judgements)
        # recall = 1 / (1 + 1) = 0.5
        assert m.abstention_recall == 0.5

    def test_judgement_distribution(self):
        items = [_item("i1"), _item("i2"), _item("i3")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.CORRECT),
            _judge("i3", Judgement.INCORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        assert m.judgement_distribution["correct"] == 2
        assert m.judgement_distribution["incorrect"] == 1
        assert m.judgement_distribution["partially_correct"] == 0

    def test_only_judged_items_count(self):
        items = [_item("i1"), _item("i2"), _item("i3")]
        judgements = [_judge("i1", Judgement.CORRECT)]
        m = compute_manual_metrics(items, judgements)
        assert m.num_judged == 1
        assert m.num_total == 3
        assert m.strict_accuracy == 1.0  # 1/1 judged

    def test_latest_judgement_wins(self):
        items = [_item("i1")]
        # load_judgements already deduplicates, so we simulate its output
        # (only the latest entry per item_id)
        judgements = [_judge("i1", Judgement.PARTIALLY_CORRECT)]
        m = compute_manual_metrics(items, judgements)
        assert m.strict_accuracy == 0.0
        assert m.lenient_accuracy == 1.0

    def test_knowledge_base_preserved(self):
        m = compute_manual_metrics([], [], knowledge_base="faq-stpo")
        assert m.knowledge_base == "faq-stpo"

    def test_abstention_metrics_zero_when_no_abstentions(self):
        items = [_item("i1"), _item("i2")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.INCORRECT),
        ]
        m = compute_manual_metrics(items, judgements)
        assert m.abstention_precision == 0.0
        assert m.abstention_recall == 0.0
