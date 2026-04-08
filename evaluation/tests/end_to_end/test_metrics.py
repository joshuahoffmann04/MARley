"""Tests for end-to-end evaluation metrics aggregation."""

from __future__ import annotations

from evaluation.end_to_end.evaluate import E2EResult
from evaluation.end_to_end.metrics import (
    E2EConfigMetrics,
    build_comparison_table,
    compute_e2e_config_metrics,
)


def _result(
    question_id: str = "q1",
    category: str = "direct",
    expected_abstention: bool = False,
    abstained: bool = False,
    abstention_level: int | None = None,
    confidence: float = 0.5,
) -> E2EResult:
    return E2EResult(
        question_id=question_id,
        question="Q?",
        reference_answer="R.",
        category=category,
        expected_abstention=expected_abstention,
        answer="" if abstained else "A.",
        abstained=abstained,
        abstention_level=abstention_level,
        abstention_reason="",
        confidence=confidence,
        retrieval_chunk_ids=["c1"],
        model="stub-model",
    )


class TestComputeE2EConfigMetrics:
    """Tests for compute_e2e_config_metrics()."""

    def test_empty_results_returns_zeros(self):
        m = compute_e2e_config_metrics([], "cfg")
        assert m.num_total == 0
        assert m.abstention_precision == 0.0
        assert m.abstention_recall == 0.0
        assert m.abstention_f1 == 0.0

    def test_correct_abstention_precision_recall(self):
        results = [
            _result("q1", expected_abstention=True, abstained=True),
            _result("q2", expected_abstention=False, abstained=True),   # incorrect
            _result("q3", expected_abstention=True, abstained=False),   # missing
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        # precision: 1 correct / (1 correct + 1 incorrect) = 0.5
        assert m.abstention_precision == 0.5
        # recall: 1 correct / (1 correct + 1 missing) = 0.5
        assert m.abstention_recall == 0.5
        # f1: 2*0.5*0.5/(0.5+0.5) = 0.5
        assert m.abstention_f1 == 0.5

    def test_all_correct_abstentions(self):
        results = [
            _result(expected_abstention=True, abstained=True),
            _result(expected_abstention=True, abstained=True),
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        assert m.abstention_precision == 1.0
        assert m.abstention_recall == 1.0
        assert m.abstention_f1 == 1.0

    def test_no_abstentions_expected_none_made(self):
        results = [
            _result(expected_abstention=False, abstained=False),
            _result(expected_abstention=False, abstained=False),
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        # Precision undefined (no abstentions made) -> 0.0
        # Recall undefined (no unanswerable questions) -> 0.0
        assert m.abstention_precision == 0.0
        assert m.abstention_recall == 0.0

    def test_abstention_rate(self):
        results = [
            _result(abstained=True),
            _result(abstained=True),
            _result(abstained=False),
            _result(abstained=False),
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        assert m.abstention_rate == 0.5
        assert m.num_abstained == 2
        assert m.num_total == 4

    def test_level1_level2_counts(self):
        results = [
            _result(abstained=True, abstention_level=1),
            _result(abstained=True, abstention_level=1),
            _result(abstained=True, abstention_level=2),
            _result(abstained=False),
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        assert m.level1_abstentions == 2
        assert m.level2_abstentions == 1

    def test_avg_confidence(self):
        results = [
            _result(confidence=0.2),
            _result(confidence=0.8),
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        assert m.avg_confidence == 0.5

    def test_per_category_breakdown(self):
        results = [
            _result("q1", category="direct", expected_abstention=False, abstained=False),
            _result("q2", category="direct", expected_abstention=False, abstained=False),
            _result("q3", category="unanswerable", expected_abstention=True, abstained=True),
        ]
        m = compute_e2e_config_metrics(results, "cfg")
        assert "direct" in m.abstention_by_category
        assert "unanswerable" in m.abstention_by_category
        assert m.abstention_by_category["unanswerable"]["num_abstained"] == 1

    def test_config_name_preserved(self):
        m = compute_e2e_config_metrics([], "my-config")
        assert m.config_name == "my-config"


class TestBuildComparisonTable:
    """Tests for build_comparison_table()."""

    def test_table_rows_match_config_count(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="a", num_total=10, num_abstained=2,
                abstention_rate=0.2, abstention_precision=0.5,
                abstention_recall=0.6, abstention_f1=0.55,
            ),
            E2EConfigMetrics(
                config_name="b", num_total=10, num_abstained=3,
                abstention_rate=0.3, abstention_precision=0.8,
                abstention_recall=0.7, abstention_f1=0.75,
            ),
        ]
        table = build_comparison_table(metrics_list)
        assert len(table) == 2

    def test_sorted_by_abstention_f1_descending(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="low", num_total=10, num_abstained=1,
                abstention_rate=0.1, abstention_precision=0.3,
                abstention_recall=0.3, abstention_f1=0.3,
            ),
            E2EConfigMetrics(
                config_name="high", num_total=10, num_abstained=5,
                abstention_rate=0.5, abstention_precision=0.9,
                abstention_recall=0.9, abstention_f1=0.9,
            ),
        ]
        table = build_comparison_table(metrics_list)
        assert table[0]["config_name"] == "high"
        assert table[1]["config_name"] == "low"

    def test_all_fields_present(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="a", num_total=10, num_abstained=2,
                abstention_rate=0.2, abstention_precision=0.5,
                abstention_recall=0.6, abstention_f1=0.55,
                avg_confidence=0.7, level1_abstentions=1, level2_abstentions=1,
            ),
        ]
        table = build_comparison_table(metrics_list)
        expected_keys = {
            "config_name", "num_total", "num_abstained", "abstention_rate",
            "abstention_precision", "abstention_recall", "abstention_f1",
            "avg_confidence", "level1_abstentions", "level2_abstentions",
        }
        assert set(table[0].keys()) == expected_keys

    def test_rounding_applied(self):
        metrics_list = [
            E2EConfigMetrics(
                config_name="a", num_total=3, num_abstained=1,
                abstention_rate=0.33333333,
                abstention_precision=0.66666666,
                abstention_recall=0.66666666,
                abstention_f1=0.66666666,
            ),
        ]
        table = build_comparison_table(metrics_list)
        assert table[0]["abstention_rate"] == 0.3333
        assert table[0]["abstention_f1"] == 0.6667
