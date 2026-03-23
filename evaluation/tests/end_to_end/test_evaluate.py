"""Tests for end-to-end evaluation runners."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.end_to_end.config import E2EConfig
from evaluation.end_to_end.evaluate import (
    E2EResult,
    load_questions,
    run_and_report,
    run_e2e_config,
    sweep_threshold,
)
from tests.conftest import KeywordRetriever, StubGenerator


# -- Fixtures --------------------------------------------------------------

CORPUS = [
    {"chunk_id": "c1", "text": "The study period is four semesters", "metadata": {}},
    {"chunk_id": "c2", "text": "Students must complete a thesis", "metadata": {}},
    {"chunk_id": "c3", "text": "The program includes a seminar", "metadata": {}},
]

QUESTIONS = [
    {
        "id": "q1",
        "question": "How long is the study period?",
        "reference_answer": "Four semesters.",
        "category": "direct",
        "expected_abstention": False,
    },
    {
        "id": "q2",
        "question": "What is the tuition fee?",
        "reference_answer": "",
        "category": "unanswerable",
        "expected_abstention": True,
    },
    {
        "id": "q3",
        "question": "Is a thesis required?",
        "reference_answer": "Yes, students must complete a thesis.",
        "category": "direct",
        "expected_abstention": False,
    },
]

CONFIG = E2EConfig(
    name="test-config",
    retriever_type="bm25",
    knowledge_bases=("stpo",),
    strategy="single",
    normalization_strategy="bm25",
    k=5,
)


def _make_retriever() -> StubRetriever:
    r = KeywordRetriever(score_multiplier=0.3)
    r.index(CORPUS)
    return r


# -- TestLoadQuestions -----------------------------------------------------


class TestLoadQuestions:
    """Tests for load_questions()."""

    def test_loads_from_json(self, tmp_path):
        data = {"questions": [{"id": "q1", "question": "test?"}]}
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_questions(str(p))
        assert len(result) == 1
        assert result[0]["id"] == "q1"

    def test_correct_field_mapping(self, tmp_path):
        data = {
            "questions": [{
                "id": "q1",
                "question": "What?",
                "reference_answer": "Answer.",
                "category": "direct",
                "expected_abstention": False,
            }]
        }
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_questions(str(p))
        assert result[0]["reference_answer"] == "Answer."
        assert result[0]["category"] == "direct"


# -- TestSweepThreshold ----------------------------------------------------


class TestSweepThreshold:
    """Tests for sweep_threshold()."""

    def test_returns_best_threshold(self):
        retriever = _make_retriever()
        best, sweep = sweep_threshold(
            retriever, QUESTIONS, "bm25",
            thresholds=[0.0, 0.3, 0.5, 0.8, 1.0],
        )
        assert isinstance(best, float)
        assert 0.0 <= best <= 1.0

    def test_sweep_covers_all_thresholds(self):
        retriever = _make_retriever()
        thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
        _, sweep = sweep_threshold(
            retriever, QUESTIONS, "bm25", thresholds=thresholds,
        )
        assert len(sweep) == len(thresholds)

    def test_threshold_zero_minimal_abstention(self):
        retriever = _make_retriever()
        _, sweep = sweep_threshold(
            retriever, QUESTIONS, "bm25", thresholds=[0.0],
        )
        # At threshold=0, nothing is filtered out (unless no results at all)
        metrics = sweep[0]["metrics"]
        total_abstentions = (
            metrics["num_correct_abstention"]
            + metrics["num_incorrect_abstention"]
        )
        assert total_abstentions <= len(QUESTIONS)

    def test_threshold_one_maximal_abstention(self):
        retriever = _make_retriever()
        _, sweep = sweep_threshold(
            retriever, QUESTIONS, "bm25", thresholds=[1.0],
        )
        # At threshold=1.0, all BM25-normalized scores < 1.0 are filtered
        metrics = sweep[0]["metrics"]
        total_abstentions = (
            metrics["num_correct_abstention"]
            + metrics["num_incorrect_abstention"]
        )
        assert total_abstentions > 0


# -- TestRunE2EConfig ------------------------------------------------------


class TestRunE2EConfig:
    """Tests for run_e2e_config()."""

    def test_answerable_question_gets_answer(self):
        results = run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            [QUESTIONS[0]], threshold=0.0,
        )
        assert len(results) == 1
        assert not results[0].abstained
        assert results[0].answer != ""

    def test_unanswerable_triggers_level1_at_high_threshold(self):
        results = run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            [QUESTIONS[1]], threshold=1.0,
        )
        assert len(results) == 1
        assert results[0].abstained
        assert results[0].abstention_level == 1

    def test_llm_abstention_triggers_level2(self):
        generator = StubGenerator(abstain_keywords={"thesis"})
        results = run_e2e_config(
            CONFIG, _make_retriever(), generator,
            [QUESTIONS[2]], threshold=0.0,
        )
        assert len(results) == 1
        assert results[0].abstained
        assert results[0].abstention_level == 2

    def test_confidence_recorded(self):
        results = run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            [QUESTIONS[0]], threshold=0.0,
        )
        assert results[0].confidence >= 0.0

    def test_retrieval_chunk_ids_recorded(self):
        results = run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            [QUESTIONS[0]], threshold=0.0,
        )
        assert len(results[0].retrieval_chunk_ids) > 0

    def test_progress_callback_called(self):
        callback = MagicMock()
        run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            QUESTIONS, threshold=0.0, progress_callback=callback,
        )
        assert callback.call_count == len(QUESTIONS)

    def test_all_questions_processed(self):
        results = run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            QUESTIONS, threshold=0.0,
        )
        assert len(results) == len(QUESTIONS)

    def test_e2e_result_fields_correct(self):
        results = run_e2e_config(
            CONFIG, _make_retriever(), StubGenerator(),
            [QUESTIONS[0]], threshold=0.0,
        )
        r = results[0]
        assert r.question_id == "q1"
        assert r.question == QUESTIONS[0]["question"]
        assert r.category == "direct"
        assert r.expected_abstention is False


# -- TestRunAndReport ------------------------------------------------------


class TestRunAndReport:
    """Tests for run_and_report()."""

    def test_report_structure_complete(self):
        report = run_and_report(
            CONFIG, _make_retriever(), StubGenerator(),
            QUESTIONS, thresholds=[0.0, 0.5, 1.0],
        )
        assert "config" in report
        assert "threshold" in report
        assert "level1_sweep" in report
        assert "abstention_metrics" in report
        assert "generator_model" in report
        assert "results" in report

    def test_threshold_from_sweep_used(self):
        report = run_and_report(
            CONFIG, _make_retriever(), StubGenerator(),
            QUESTIONS, thresholds=[0.0, 0.5, 1.0],
        )
        assert report["threshold"] in [0.0, 0.5, 1.0]

    def test_abstention_metrics_included(self):
        report = run_and_report(
            CONFIG, _make_retriever(), StubGenerator(),
            QUESTIONS, thresholds=[0.0, 0.5, 1.0],
        )
        am = report["abstention_metrics"]
        assert "precision" in am
        assert "recall" in am
        assert "f1" in am
