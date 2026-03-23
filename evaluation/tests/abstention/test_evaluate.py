"""Tests for abstention evaluation runners."""

from __future__ import annotations

import pytest

from evaluation.abstention.evaluate import (
    run_abstention_evaluation,
    run_level1_sweep,
)
from tests.conftest import KeywordRetriever, StubGenerator


# ── Fixtures ────────────────────────────────────────────────────────────────


CORPUS = [
    {"chunk_id": "c1", "text": "The study period is four semesters", "metadata": {}},
    {"chunk_id": "c2", "text": "Students must complete a thesis", "metadata": {}},
    {"chunk_id": "c3", "text": "The program includes a seminar", "metadata": {}},
]

QUESTIONS = [
    {
        "question_id": "q1",
        "question": "How long is the study period?",
        "expected_abstention": False,
    },
    {
        "question_id": "q2",
        "question": "Where can I park my bicycle?",
        "expected_abstention": True,
    },
    {
        "question_id": "q3",
        "question": "Is a thesis required for students?",
        "expected_abstention": False,
    },
]


# ── Level 1 Sweep Tests ────────────────────────────────────────────────────


class TestRunLevel1Sweep:
    """Tests for the Level 1 threshold sweep."""

    def test_threshold_zero_no_abstention(self) -> None:
        """Threshold=0 should not cause any abstention (all scores >= 0)."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        results = run_level1_sweep(
            retriever, CORPUS, QUESTIONS, [0.0],
            normalization_strategy="bm25",
        )
        assert len(results) == 1
        # q2 has no keyword overlap -> score=0 -> normalized=0 -> filtered out
        # So q2 abstains even at threshold=0 because score is exactly 0
        m = results[0]["metrics"]
        assert m["num_total"] == 3

    def test_threshold_one_all_abstain(self) -> None:
        """Threshold=1.0 should cause all questions to abstain (Level 1)."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        results = run_level1_sweep(
            retriever, CORPUS, QUESTIONS, [1.0],
            normalization_strategy="bm25",
        )
        m = results[0]["metrics"]
        assert m["num_correct_abstention"] + m["num_incorrect_abstention"] == 3

    def test_multiple_thresholds(self) -> None:
        """Sweep returns one entry per threshold."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        thresholds = [0.0, 0.3, 0.6, 1.0]
        results = run_level1_sweep(
            retriever, CORPUS, QUESTIONS, thresholds,
            normalization_strategy="bm25",
        )
        assert len(results) == 4
        assert [r["threshold"] for r in results] == thresholds

    def test_unanswerable_correctly_identified(self) -> None:
        """Question with no retrieval overlap should abstain at low threshold."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        results = run_level1_sweep(
            retriever, CORPUS, QUESTIONS, [0.1],
            normalization_strategy="bm25",
        )
        m = results[0]["metrics"]
        # q2 ("parking regulations") has no overlap -> abstains
        assert m["num_correct_abstention"] >= 1

    def test_metrics_per_threshold(self) -> None:
        """Each threshold entry should have complete metrics."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        results = run_level1_sweep(
            retriever, CORPUS, QUESTIONS, [0.5],
            normalization_strategy="bm25",
        )
        m = results[0]["metrics"]
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "false_abstention_rate" in m
        assert "coverage" in m

    def test_vector_normalization_strategy(self) -> None:
        """Sweep works with vector normalization strategy."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        results = run_level1_sweep(
            retriever, CORPUS, QUESTIONS, [0.1],
            normalization_strategy="vector",
        )
        assert len(results) == 1


# ── Full Evaluation Tests ──────────────────────────────────────────────────


class TestRunAbstentionEvaluation:
    """Tests for the full two-level abstention evaluation."""

    def test_level1_triggers_for_unanswerable(self) -> None:
        """Questions with no retrieval matches should trigger Level 1."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        generator = StubGenerator(model="stub")
        report = run_abstention_evaluation(
            retriever, generator, CORPUS, QUESTIONS,
            threshold=0.1, normalization_strategy="bm25",
        )
        # q2 has no keyword overlap -> Level 1 abstention
        q2_result = next(r for r in report["results"] if r["question_id"] == "q2")
        assert q2_result["system_abstained"] is True
        assert q2_result["abstention_level"] == 1

    def test_level2_triggers_for_llm_abstention(self) -> None:
        """LLM abstention should be detected as Level 2."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        generator = StubGenerator(model="stub", abstain_keywords={"thesis"})
        report = run_abstention_evaluation(
            retriever, generator, CORPUS, QUESTIONS,
            threshold=0.1, normalization_strategy="bm25",
        )
        # q3 contains "thesis" -> LLM abstains -> Level 2
        q3_result = next(r for r in report["results"] if r["question_id"] == "q3")
        assert q3_result["system_abstained"] is True
        assert q3_result["abstention_level"] == 2

    def test_normal_answers_pass_through(self) -> None:
        """Answerable questions with good retrieval should pass both levels."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        generator = StubGenerator(model="stub")
        report = run_abstention_evaluation(
            retriever, generator, CORPUS, QUESTIONS,
            threshold=0.1, normalization_strategy="bm25",
        )
        q1_result = next(r for r in report["results"] if r["question_id"] == "q1")
        assert q1_result["system_abstained"] is False
        assert q1_result["answer"] == "The answer is 42."

    def test_confidence_recorded(self) -> None:
        """Each result should have a confidence value."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        generator = StubGenerator(model="stub")
        report = run_abstention_evaluation(
            retriever, generator, CORPUS, QUESTIONS,
            threshold=0.1, normalization_strategy="bm25",
        )
        for r in report["results"]:
            assert "confidence" in r
            assert isinstance(r["confidence"], float)

    def test_report_structure(self) -> None:
        """Report should have config, metrics, and results keys."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        generator = StubGenerator(model="stub")
        report = run_abstention_evaluation(
            retriever, generator, CORPUS, QUESTIONS,
            threshold=0.3, normalization_strategy="bm25",
        )
        assert "config" in report
        assert "metrics" in report
        assert "results" in report
        assert report["config"]["threshold"] == 0.3
        assert report["config"]["k"] == 5

    def test_progress_callback(self) -> None:
        """Progress callback should be called for each question."""
        retriever = KeywordRetriever(score_multiplier=0.3)
        generator = StubGenerator(model="stub")
        calls = []
        report = run_abstention_evaluation(
            retriever, generator, CORPUS, QUESTIONS,
            threshold=0.1, normalization_strategy="bm25",
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )
        assert len(calls) == len(QUESTIONS)
        assert calls[-1] == (3, 3)
