"""Tests for the evaluation runner module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path as P
from unittest.mock import MagicMock

import pytest

from evaluation.retrieval.evaluate import load_evaluation, run_and_report, run_evaluation
from evaluation.retrieval.metrics import RetrievalMetrics
from src.marley.retrieval.base import RetrievalResult, Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_eval_json(path: P, questions: list[dict]) -> None:
    """Write a minimal evaluation JSON file."""
    data = {
        "metadata": {"version": "1.0", "knowledge_base": "test"},
        "questions": questions,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_questions() -> list[dict]:
    """Return a small set of evaluation questions for testing."""
    return [
        {
            "id": "eval-001",
            "question": "How long is the standard study period?",
            "relevant_chunks": ["par-7-txt-1"],
            "category": "direct",
            "expected_abstention": False,
        },
        {
            "id": "eval-002",
            "question": "What degree title do graduates receive?",
            "relevant_chunks": ["par-3-txt-1"],
            "category": "direct",
            "expected_abstention": False,
        },
        {
            "id": "eval-076",
            "question": "What is the tuition fee?",
            "relevant_chunks": [],
            "category": "unanswerable",
            "expected_abstention": True,
        },
    ]


class _StubRetriever(Retriever):
    """Retriever that returns predefined results for testing."""

    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = responses
        self._corpus_size = 10

    def index(self, corpus: list[dict]) -> None:
        self._corpus_size = len(corpus)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        ids = self._responses.get(query, [])[:k]
        return [
            RetrievalResult(chunk_id=cid, text="text", score=1.0, metadata={})
            for cid in ids
        ]

    @property
    def size(self) -> int:
        return self._corpus_size


# ---------------------------------------------------------------------------
# TestLoadEvaluation
# ---------------------------------------------------------------------------

class TestLoadEvaluation:
    def test_loads_questions(self, tmp_path):
        path = tmp_path / "eval.json"
        questions = _make_questions()
        _write_eval_json(path, questions)

        loaded = load_evaluation(path)
        assert len(loaded) == 3
        assert loaded[0]["id"] == "eval-001"

    def test_returns_list_of_dicts(self, tmp_path):
        path = tmp_path / "eval.json"
        _write_eval_json(path, _make_questions())

        loaded = load_evaluation(path)
        assert isinstance(loaded, list)
        assert all(isinstance(q, dict) for q in loaded)

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "eval.json"
        _write_eval_json(path, _make_questions())

        loaded = load_evaluation(str(path))
        assert len(loaded) == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_evaluation("/nonexistent/eval.json")


# ---------------------------------------------------------------------------
# TestRunEvaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_skips_unanswerable(self):
        retriever = _StubRetriever({})
        questions = _make_questions()

        metrics = run_evaluation(retriever, questions, k=5)
        # 3 questions total, 1 unanswerable skipped -> 2 evaluated
        assert metrics.num_queries == 2

    def test_skips_empty_relevant_chunks(self):
        questions = [
            {
                "id": "q1",
                "question": "test?",
                "relevant_chunks": [],
                "expected_abstention": False,
            },
        ]
        retriever = _StubRetriever({})
        metrics = run_evaluation(retriever, questions, k=5)
        assert metrics.num_queries == 0

    def test_perfect_retrieval(self):
        retriever = _StubRetriever({
            "How long is the standard study period?": ["par-7-txt-1"],
            "What degree title do graduates receive?": ["par-3-txt-1"],
        })
        questions = _make_questions()

        metrics = run_evaluation(retriever, questions, k=5)
        assert metrics.num_queries == 2
        assert metrics.recall_at_k == 1.0
        assert metrics.mrr == 1.0

    def test_no_hits(self):
        retriever = _StubRetriever({})
        questions = _make_questions()

        metrics = run_evaluation(retriever, questions, k=5)
        assert metrics.precision_at_k == 0.0
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0

    def test_returns_retrieval_metrics(self):
        retriever = _StubRetriever({})
        metrics = run_evaluation(retriever, _make_questions(), k=5)
        assert isinstance(metrics, RetrievalMetrics)

    def test_skip_unanswerable_false(self):
        questions = [
            {
                "id": "q1",
                "question": "unanswerable?",
                "relevant_chunks": ["c1"],
                "expected_abstention": True,
            },
        ]
        retriever = _StubRetriever({"unanswerable?": ["c1"]})

        # With skip_unanswerable=False, unanswerable questions ARE evaluated
        metrics = run_evaluation(
            retriever, questions, k=5, skip_unanswerable=False,
        )
        assert metrics.num_queries == 1


# ---------------------------------------------------------------------------
# TestRunAndReport
# ---------------------------------------------------------------------------

class TestRunAndReport:
    def test_report_structure(self, tmp_path):
        path = tmp_path / "eval.json"
        _write_eval_json(path, _make_questions())

        retriever = _StubRetriever({})
        report = run_and_report(retriever, path, k=5)

        assert "eval_file" in report
        assert "metrics" in report
        assert "config" in report
        assert report["config"]["k"] == 5
        assert report["config"]["retriever_type"] == "_StubRetriever"

    def test_report_metrics_are_dict(self, tmp_path):
        path = tmp_path / "eval.json"
        _write_eval_json(path, _make_questions())

        retriever = _StubRetriever({})
        report = run_and_report(retriever, path, k=5)

        assert isinstance(report["metrics"], dict)
        assert "precision_at_k" in report["metrics"]
        assert "recall_at_k" in report["metrics"]
        assert "mrr" in report["metrics"]

    def test_report_corpus_size(self, tmp_path):
        path = tmp_path / "eval.json"
        _write_eval_json(path, _make_questions())

        retriever = _StubRetriever({})
        report = run_and_report(retriever, path, k=5)

        assert report["config"]["corpus_size"] == 10
