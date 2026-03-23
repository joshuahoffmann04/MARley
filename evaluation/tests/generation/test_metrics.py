"""Tests for generation evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.generation.metrics import (
    GenerationEvalResult,
    GenerationMetrics,
    compute_generation_metrics,
)


def _make_result(qid: str, n_dist: int) -> GenerationEvalResult:
    """Create a minimal GenerationEvalResult for testing."""
    return GenerationEvalResult(
        question_id=qid,
        num_distractors=n_dist,
        generated_answer="ans",
        reference_answer="ref",
    )


class TestComputeGenerationMetrics:
    """Tests for compute_generation_metrics()."""

    def test_empty_results(self):
        m = compute_generation_metrics([], "stpo", "llama")
        assert m.num_results == 0
        assert m.num_queries == 0
        assert m.results_by_distractors == {}

    def test_single_result(self):
        results = [_make_result("q1", 0)]
        m = compute_generation_metrics(results, "stpo", "llama")
        assert m.num_results == 1
        assert m.num_queries == 1
        assert m.results_by_distractors == {0: 1}

    def test_multiple_levels(self):
        results = [
            _make_result("q1", 0),
            _make_result("q1", 1),
            _make_result("q2", 0),
            _make_result("q2", 1),
        ]
        m = compute_generation_metrics(results, "stpo", "llama")
        assert m.num_results == 4
        assert m.num_queries == 2
        assert m.results_by_distractors[0] == 2
        assert m.results_by_distractors[1] == 2

    def test_metadata_fields(self):
        results = [_make_result("q1", 0)]
        m = compute_generation_metrics(results, "faq-ao", "mistral")
        assert m.knowledge_base == "faq-ao"
        assert m.model == "mistral"

    def test_distractor_levels_sorted(self):
        results = [
            _make_result("q1", 10),
            _make_result("q1", 0),
            _make_result("q1", 5),
        ]
        m = compute_generation_metrics(results, "stpo", "llama")
        levels = list(m.results_by_distractors.keys())
        assert levels == [0, 5, 10]
