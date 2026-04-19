"""Tests for evaluation/utils.py shared utilities."""

from __future__ import annotations

import json

import pytest

from evaluation.utils import (
    AbstentionMetrics,
    compute_abstention_metrics,
    load_evaluation,
    load_json,
    merge_chunks,
    merge_evaluation_data,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _chunk_file(chunks):
    return {"chunks": chunks}


def _eval_file(questions):
    return {"metadata": {"version": "1.0"}, "questions": questions}


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_valid_json(self, tmp_path):
        p = tmp_path / "data.json"
        _write_json(p, {"key": "value"})
        assert load_json(p) == {"key": "value"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# load_evaluation
# ---------------------------------------------------------------------------


class TestLoadEvaluation:
    def test_returns_questions_list(self, tmp_path):
        p = tmp_path / "eval.json"
        q = [{"id": "q1", "question": "What?", "relevant_chunks": ["c1"]}]
        _write_json(p, _eval_file(q))
        result = load_evaluation(p)
        assert len(result) == 1
        assert result[0]["id"] == "q1"

    def test_validates_structure(self, tmp_path):
        p = tmp_path / "eval.json"
        q = [{"id": "q1", "question": "What?", "relevant_chunks": ["c1"],
              "category": "direct", "expected_abstention": False}]
        _write_json(p, _eval_file(q))
        result = load_evaluation(p)
        assert result[0]["category"] == "direct"
        assert result[0]["expected_abstention"] is False

    def test_empty_questions(self, tmp_path):
        p = tmp_path / "eval.json"
        _write_json(p, _eval_file([]))
        assert load_evaluation(p) == []


# ---------------------------------------------------------------------------
# merge_chunks
# ---------------------------------------------------------------------------


class TestMergeChunks:
    def test_single_file(self, tmp_path):
        p = tmp_path / "chunks.json"
        chunks = [{"chunk_id": "c1", "text": "hello", "metadata": {}}]
        _write_json(p, _chunk_file(chunks))
        result = merge_chunks(p)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_two_files_concatenated(self, tmp_path):
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        _write_json(p1, _chunk_file([{"chunk_id": "c1", "text": "a", "metadata": {}}]))
        _write_json(p2, _chunk_file([{"chunk_id": "c2", "text": "b", "metadata": {}}]))
        result = merge_chunks(p1, p2)
        assert len(result) == 2
        assert [c["chunk_id"] for c in result] == ["c1", "c2"]

    def test_duplicate_chunk_id_raises(self, tmp_path):
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        _write_json(p1, _chunk_file([{"chunk_id": "c1", "text": "a", "metadata": {}}]))
        _write_json(p2, _chunk_file([{"chunk_id": "c1", "text": "b", "metadata": {}}]))
        with pytest.raises(ValueError, match="Duplicate chunk_id"):
            merge_chunks(p1, p2)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.json"
        _write_json(p, _chunk_file([]))
        assert merge_chunks(p) == []


# ---------------------------------------------------------------------------
# merge_evaluation_data
# ---------------------------------------------------------------------------


class TestMergeEvaluationData:
    def test_single_kb(self, tmp_path):
        p = tmp_path / "eval.json"
        q = [{"id": "q1", "question": "What?", "relevant_chunks": ["c1"],
              "reference_answer": "A.", "category": "direct",
              "expected_abstention": False}]
        _write_json(p, _eval_file(q))
        result = merge_evaluation_data({"kb1": p})
        assert len(result) == 1
        assert result[0]["relevant_chunks"] == ["c1"]

    def test_two_kbs_merge_relevant_chunks(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        p2 = tmp_path / "eval2.json"
        _write_json(p1, _eval_file([
            {"id": "q1", "question": "What?", "relevant_chunks": ["c1"],
             "reference_answer": "A.", "category": "direct",
             "expected_abstention": False},
        ]))
        _write_json(p2, _eval_file([
            {"id": "q1", "question": "What?", "relevant_chunks": ["c2"],
             "reference_answer": "A.", "category": "direct",
             "expected_abstention": False},
        ]))
        result = merge_evaluation_data({"kb1": p1, "kb2": p2})
        assert len(result) == 1
        assert sorted(result[0]["relevant_chunks"]) == ["c1", "c2"]

    def test_question_in_one_kb_only(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        p2 = tmp_path / "eval2.json"
        _write_json(p1, _eval_file([
            {"id": "q1", "question": "What?", "relevant_chunks": ["c1"],
             "reference_answer": "A.", "category": "direct",
             "expected_abstention": False},
        ]))
        _write_json(p2, _eval_file([
            {"id": "q2", "question": "Why?", "relevant_chunks": ["c2"],
             "reference_answer": "B.", "category": "direct",
             "expected_abstention": False},
        ]))
        result = merge_evaluation_data({"kb1": p1, "kb2": p2})
        assert len(result) == 2
        ids = {q["id"] for q in result}
        assert ids == {"q1", "q2"}


# ---------------------------------------------------------------------------
# compute_abstention_metrics
# ---------------------------------------------------------------------------


class TestComputeAbstentionMetrics:
    def test_perfect_classification(self):
        results = [
            {"expected_abstention": True, "system_abstained": True},
            {"expected_abstention": False, "system_abstained": False},
        ]
        m = compute_abstention_metrics(results, threshold=0.5)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.false_abstention_rate == pytest.approx(0.0)
        assert m.num_total == 2

    def test_all_false_positives(self):
        results = [
            {"expected_abstention": False, "system_abstained": True},
            {"expected_abstention": False, "system_abstained": True},
        ]
        m = compute_abstention_metrics(results, threshold=0.5)
        assert m.precision == pytest.approx(0.0)
        assert m.num_incorrect_abstention == 2

    def test_empty_input(self):
        m = compute_abstention_metrics([], threshold=0.5)
        assert m.num_total == 0
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_mixed_results(self):
        results = [
            {"expected_abstention": True, "system_abstained": True},
            {"expected_abstention": True, "system_abstained": False},
            {"expected_abstention": False, "system_abstained": True},
            {"expected_abstention": False, "system_abstained": False},
        ]
        m = compute_abstention_metrics(results, threshold=0.3)
        assert m.num_correct_abstention == 1
        assert m.num_incorrect_abstention == 1
        assert m.num_missing_abstention == 1
        assert m.num_answered == 1
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(0.5)
        assert m.false_abstention_rate == pytest.approx(0.5)
        assert m.coverage == pytest.approx(0.5)
        assert m.threshold == 0.3


class TestAbstentionMetricsF0_5:
    """Phase 12: F0.5 is computed and stored alongside F1.

    F_beta = (1+b^2) * P * R / (b^2 * P + R); beta=0.5 weights precision 2x.
    """

    def test_f0_5_equals_precision_when_recall_matches(self):
        # P = R = 0.5 -> F0.5 = F1 = 0.5
        results = [
            {"expected_abstention": True, "system_abstained": True},
            {"expected_abstention": True, "system_abstained": False},
            {"expected_abstention": False, "system_abstained": True},
            {"expected_abstention": False, "system_abstained": False},
        ]
        m = compute_abstention_metrics(results, threshold=0.3)
        assert m.f0_5 == pytest.approx(0.5)

    def test_f0_5_favours_precision_over_recall(self):
        # 4 correct abstentions + 4 missing -> P=1.0, R=0.5
        # F1 = 2*1*0.5/(1+0.5) = 0.667
        # F0.5 = 1.25*1*0.5/(0.25*1+0.5) = 0.833
        results = [
            {"expected_abstention": True, "system_abstained": True}
            for _ in range(4)
        ] + [
            {"expected_abstention": True, "system_abstained": False}
            for _ in range(4)
        ]
        m = compute_abstention_metrics(results, threshold=0.5)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))
        assert m.f0_5 == pytest.approx(1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5))
        assert m.f0_5 > m.f1  # precision-weighted metric rewards precision

    def test_f0_5_penalises_low_precision(self):
        # Low precision: 1 correct + many false abstentions
        # P=1/5=0.2, R=1.0
        # F1 = 2*0.2*1/1.2 = 0.333
        # F0.5 = 1.25*0.2*1/(0.05+1) = 0.238
        results = [
            {"expected_abstention": True, "system_abstained": True}
        ] + [
            {"expected_abstention": False, "system_abstained": True}
            for _ in range(4)
        ]
        m = compute_abstention_metrics(results, threshold=0.5)
        assert m.f0_5 < m.f1  # when P < R, F0.5 is stricter than F1

    def test_f0_5_zero_when_precision_and_recall_zero(self):
        # Neither abstention nor recall: system never abstains on unanswerable
        results = [
            {"expected_abstention": True, "system_abstained": False}
            for _ in range(3)
        ]
        m = compute_abstention_metrics(results, threshold=0.0)
        assert m.precision == pytest.approx(1.0)  # no predictions
        assert m.recall == pytest.approx(0.0)
        assert m.f0_5 == pytest.approx(0.0)
