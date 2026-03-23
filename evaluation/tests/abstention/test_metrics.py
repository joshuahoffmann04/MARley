"""Tests for abstention evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.abstention.metrics import AbstentionMetrics, compute_abstention_metrics


def _r(qid: str, expected: bool, abstained: bool, level: int | None = None) -> dict:
    """Helper to create a result dict."""
    return {
        "question_id": qid,
        "expected_abstention": expected,
        "system_abstained": abstained,
        "abstention_level": level,
    }


class TestComputeAbstentionMetrics:
    """Tests for compute_abstention_metrics."""

    def test_perfect_abstention(self) -> None:
        """System abstains on all unanswerable, answers all answerable."""
        results = [
            _r("q1", True, True, 1),
            _r("q2", True, True, 2),
            _r("q3", False, False),
            _r("q4", False, False),
        ]
        m = compute_abstention_metrics(results, threshold=0.3)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.false_abstention_rate == pytest.approx(0.0)
        assert m.coverage == pytest.approx(0.5)

    def test_no_abstentions(self) -> None:
        """System never abstains — recall is 0 for unanswerable."""
        results = [
            _r("q1", True, False),
            _r("q2", False, False),
        ]
        m = compute_abstention_metrics(results, threshold=0.0)
        assert m.precision == pytest.approx(1.0)  # 0/0 -> 1.0
        assert m.recall == pytest.approx(0.0)
        assert m.num_missing_abstention == 1
        assert m.coverage == pytest.approx(1.0)

    def test_all_abstain(self) -> None:
        """System abstains on everything — coverage is 0."""
        results = [
            _r("q1", True, True, 1),
            _r("q2", False, True, 1),
        ]
        m = compute_abstention_metrics(results, threshold=0.9)
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(1.0)
        assert m.coverage == pytest.approx(0.0)
        assert m.false_abstention_rate == pytest.approx(1.0)

    def test_mixed_results(self) -> None:
        """Typical mixed scenario."""
        results = [
            _r("q1", True, True, 1),    # correct abstention
            _r("q2", True, False),       # missing abstention
            _r("q3", False, True, 2),    # incorrect abstention
            _r("q4", False, False),      # correct answer
            _r("q5", False, False),      # correct answer
        ]
        m = compute_abstention_metrics(results, threshold=0.3)
        assert m.num_correct_abstention == 1
        assert m.num_incorrect_abstention == 1
        assert m.num_missing_abstention == 1
        assert m.num_answered == 2
        assert m.precision == pytest.approx(0.5)    # 1/2
        assert m.recall == pytest.approx(0.5)       # 1/2
        assert m.false_abstention_rate == pytest.approx(1 / 3)  # 1/(2+1)

    def test_f1_computation(self) -> None:
        """Verify F1 is the harmonic mean."""
        results = [
            _r("q1", True, True, 1),
            _r("q2", True, False),
            _r("q3", False, True, 1),
            _r("q4", False, False),
        ]
        m = compute_abstention_metrics(results, threshold=0.3)
        # P = 1/2, R = 1/2, F1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert m.f1 == pytest.approx(0.5)

    def test_false_abstention_rate(self) -> None:
        """False abstention rate = incorrect / answerable."""
        results = [
            _r("q1", False, True, 1),   # incorrect
            _r("q2", False, True, 1),   # incorrect
            _r("q3", False, False),     # correct answer
        ]
        m = compute_abstention_metrics(results, threshold=0.5)
        assert m.false_abstention_rate == pytest.approx(2 / 3)

    def test_empty_results(self) -> None:
        """No results -> default metrics."""
        m = compute_abstention_metrics([], threshold=0.3)
        assert m.num_total == 0
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)

    def test_single_answerable(self) -> None:
        """Single answerable question, correctly answered."""
        m = compute_abstention_metrics([_r("q1", False, False)], threshold=0.3)
        assert m.num_answered == 1
        assert m.coverage == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)  # no unanswerable -> 1.0

    def test_single_unanswerable(self) -> None:
        """Single unanswerable question, correctly abstained."""
        m = compute_abstention_metrics([_r("q1", True, True, 1)], threshold=0.3)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.coverage == pytest.approx(0.0)

    def test_threshold_stored(self) -> None:
        """Threshold value should be stored in the metrics."""
        m = compute_abstention_metrics([_r("q1", False, False)], threshold=0.42)
        assert m.threshold == pytest.approx(0.42)
