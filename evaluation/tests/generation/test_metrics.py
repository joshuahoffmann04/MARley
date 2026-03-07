"""Tests for generation evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.generation.metrics import (
    GenerationEvalResult,
    GenerationMetrics,
    compute_generation_metrics,
)


def _make_result(qid: str, n_dist: int, correct: bool) -> GenerationEvalResult:
    """Create a minimal GenerationEvalResult for testing."""
    return GenerationEvalResult(
        question_id=qid,
        num_distractors=n_dist,
        correct=correct,
        confidence=0.9,
        generated_answer="ans",
        reference_answer="ref",
        judge_reasoning="ok",
    )


class TestComputeGenerationMetrics:
    """Tests for compute_generation_metrics()."""

    def test_empty_results(self):
        m = compute_generation_metrics([], "stpo", "llama", "llama")
        assert m.accuracy == 0.0
        assert m.num_queries == 0
        assert m.accuracy_by_distractors == {}

    def test_all_correct(self):
        results = [
            _make_result("q1", 0, True),
            _make_result("q1", 1, True),
            _make_result("q2", 0, True),
            _make_result("q2", 1, True),
        ]
        m = compute_generation_metrics(results, "stpo", "llama", "llama")
        assert m.accuracy == 1.0
        assert m.accuracy_by_distractors[0] == 1.0
        assert m.accuracy_by_distractors[1] == 1.0
        assert m.num_queries == 2

    def test_mixed_correctness(self):
        results = [
            _make_result("q1", 0, True),
            _make_result("q1", 5, False),
            _make_result("q2", 0, True),
            _make_result("q2", 5, True),
        ]
        m = compute_generation_metrics(results, "stpo", "llama", "llama")
        assert m.accuracy == 0.75
        assert m.accuracy_by_distractors[0] == 1.0
        assert m.accuracy_by_distractors[5] == 0.5

    def test_all_incorrect(self):
        results = [_make_result("q1", 0, False), _make_result("q1", 1, False)]
        m = compute_generation_metrics(results, "stpo", "llama", "llama")
        assert m.accuracy == 0.0

    def test_metadata_fields(self):
        results = [_make_result("q1", 0, True)]
        m = compute_generation_metrics(results, "faq-ao", "mistral", "llama")
        assert m.knowledge_base == "faq-ao"
        assert m.model == "mistral"
        assert m.judge_model == "llama"

    def test_distractor_levels_sorted(self):
        results = [
            _make_result("q1", 10, True),
            _make_result("q1", 0, True),
            _make_result("q1", 5, False),
        ]
        m = compute_generation_metrics(results, "stpo", "llama", "llama")
        levels = list(m.accuracy_by_distractors.keys())
        assert levels == [0, 5, 10]
