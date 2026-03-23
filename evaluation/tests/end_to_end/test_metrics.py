"""Tests for end-to-end evaluation metrics aggregation."""

from __future__ import annotations

from evaluation.end_to_end.metrics import (
    E2EConfigMetrics,
    build_comparison_table,
    compute_e2e_config_metrics,
)
from evaluation.manual.models import (
    EvaluationItem,
    Judgement,
    ManualJudgement,
)


def _item(
    item_id: str,
    category: str = "direct",
    expected_abstention: bool = False,
) -> EvaluationItem:
    return EvaluationItem(
        id=item_id,
        question="Q?",
        generated_answer="A.",
        reference_answer="R.",
        category=category,
        expected_abstention=expected_abstention,
    )


def _judge(item_id: str, judgement: Judgement) -> ManualJudgement:
    return ManualJudgement(
        item_id=item_id,
        judgement=judgement,
    )


class TestComputeE2EConfigMetrics:
    """Tests for compute_e2e_config_metrics()."""

    def test_all_correct_accuracy_is_one(self):
        items = [_item("i1"), _item("i2"), _item("i3")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.CORRECT),
            _judge("i3", Judgement.CORRECT),
        ]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        assert m.strict_accuracy == 1.0
        assert m.lenient_accuracy == 1.0

    def test_mixed_judgements(self):
        items = [_item("i1"), _item("i2"), _item("i3"), _item("i4")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.PARTIALLY_CORRECT),
            _judge("i3", Judgement.INCORRECT),
            _judge("i4", Judgement.INCORRECT),
        ]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        # strict: 1/4 = 0.25
        assert m.strict_accuracy == 0.25
        # lenient: 2/4 = 0.5
        assert m.lenient_accuracy == 0.5

    def test_accuracy_by_category(self):
        items = [
            _item("i1", category="direct"),
            _item("i2", category="direct"),
            _item("i3", category="multi_source"),
        ]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.INCORRECT),
            _judge("i3", Judgement.CORRECT),
        ]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        assert m.strict_accuracy_by_category["direct"] == 0.5
        assert m.strict_accuracy_by_category["multi_source"] == 1.0

    def test_abstention_precision_recall_f1(self):
        items = [
            _item("i1", expected_abstention=True),
            _item("i2", expected_abstention=True),
            _item("i3", expected_abstention=False),
        ]
        judgements = [
            _judge("i1", Judgement.CORRECT_ABSTENTION),
            _judge("i2", Judgement.MISSING_ABSTENTION),
            _judge("i3", Judgement.INCORRECT_ABSTENTION),
        ]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        # precision: 1 correct / (1 correct + 1 incorrect) = 0.5
        assert m.abstention_precision == 0.5
        # recall: 1 correct / (1 correct + 1 missing) = 0.5
        assert m.abstention_recall == 0.5
        # f1: 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert m.abstention_f1 == 0.5

    def test_empty_judgements_returns_zeros(self):
        items = [_item("i1")]
        m = compute_e2e_config_metrics(items, [], "cfg")
        assert m.strict_accuracy == 0.0
        assert m.lenient_accuracy == 0.0
        assert m.num_judged == 0
        assert m.num_total == 1

    def test_correct_abstention_counted_in_accuracy(self):
        items = [
            _item("i1"),
            _item("i2", category="unanswerable", expected_abstention=True),
        ]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.CORRECT_ABSTENTION),
        ]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        # Both correct: 2/2 = 1.0
        assert m.strict_accuracy == 1.0
        assert m.lenient_accuracy == 1.0

    def test_partial_correctness_in_lenient_only(self):
        items = [_item("i1")]
        judgements = [_judge("i1", Judgement.PARTIALLY_CORRECT)]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        assert m.strict_accuracy == 0.0
        assert m.lenient_accuracy == 1.0

    def test_judgement_distribution(self):
        items = [_item("i1"), _item("i2")]
        judgements = [
            _judge("i1", Judgement.CORRECT),
            _judge("i2", Judgement.INCORRECT),
        ]
        m = compute_e2e_config_metrics(items, judgements, "cfg")
        assert m.judgement_distribution["correct"] == 1
        assert m.judgement_distribution["incorrect"] == 1


class TestBuildComparisonTable:
    """Tests for build_comparison_table()."""

    def test_table_rows_match_config_count(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="a", strict_accuracy=0.5, lenient_accuracy=0.6,
            ),
            E2EConfigMetrics(
                config_name="b", strict_accuracy=0.7, lenient_accuracy=0.8,
            ),
        ]
        table = build_comparison_table(metrics_list)
        assert len(table) == 2

    def test_sorted_by_lenient_accuracy_descending(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="low", strict_accuracy=0.3, lenient_accuracy=0.4,
            ),
            E2EConfigMetrics(
                config_name="high", strict_accuracy=0.9, lenient_accuracy=0.95,
            ),
            E2EConfigMetrics(
                config_name="mid", strict_accuracy=0.6, lenient_accuracy=0.7,
            ),
        ]
        table = build_comparison_table(metrics_list)
        assert table[0]["config_name"] == "high"
        assert table[1]["config_name"] == "mid"
        assert table[2]["config_name"] == "low"

    def test_all_fields_present(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="a", strict_accuracy=0.5, lenient_accuracy=0.6,
                strict_accuracy_by_category={
                    "direct": 0.5, "multi_source": 0.4, "unanswerable": 0.6,
                },
                abstention_precision=0.8, abstention_recall=0.7, abstention_f1=0.75,
                num_judged=10,
            ),
        ]
        table = build_comparison_table(metrics_list)
        row = table[0]
        expected_keys = {
            "config_name", "strict_accuracy", "lenient_accuracy",
            "strict_direct", "strict_multi_source", "strict_unanswerable",
            "abstention_precision", "abstention_recall", "abstention_f1",
            "num_judged",
        }
        assert set(row.keys()) == expected_keys

    def test_rounding_applied(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="a",
                strict_accuracy=0.33333333,
                lenient_accuracy=0.66666666,
            ),
        ]
        table = build_comparison_table(metrics_list)
        assert table[0]["strict_accuracy"] == 0.3333
        assert table[0]["lenient_accuracy"] == 0.6667
