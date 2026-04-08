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


class TestGenerationEvalResultQualityFields:
    """Tests for the quality metric fields of GenerationEvalResult."""

    def test_default_quality_fields_are_zero(self):
        r = _make_result("q1", 0)
        assert r.rouge1 == 0.0
        assert r.rouge2 == 0.0
        assert r.rougeL == 0.0
        assert r.bertscore_f1 == 0.0
        assert r.faithfulness == 0.0
        assert r.answer_relevance == 0.0
        assert r.correctness == 0.0

    def test_quality_fields_settable(self):
        r = GenerationEvalResult(
            question_id="q1",
            num_distractors=0,
            generated_answer="ans",
            reference_answer="ref",
            rouge1=0.8,
            rouge2=0.5,
            rougeL=0.7,
            bertscore_f1=0.9,
            faithfulness=0.95,
            answer_relevance=0.85,
            correctness=0.80,
        )
        assert r.rouge1 == pytest.approx(0.8)
        assert r.rouge2 == pytest.approx(0.5)
        assert r.rougeL == pytest.approx(0.7)
        assert r.bertscore_f1 == pytest.approx(0.9)
        assert r.faithfulness == pytest.approx(0.95)
        assert r.answer_relevance == pytest.approx(0.85)
        assert r.correctness == pytest.approx(0.80)


class TestComputeGenerationMetricsQuality:
    """Tests for quality score aggregation in compute_generation_metrics."""

    def _make_result_with_scores(
        self,
        qid: str,
        n_dist: int,
        rouge1: float = 0.0,
        rouge2: float = 0.0,
        rougeL: float = 0.0,
        bertscore_f1: float = 0.0,
        faithfulness: float = 0.0,
        answer_relevance: float = 0.0,
        correctness: float = 0.0,
    ) -> GenerationEvalResult:
        return GenerationEvalResult(
            question_id=qid,
            num_distractors=n_dist,
            generated_answer="ans",
            reference_answer="ref",
            rouge1=rouge1,
            rouge2=rouge2,
            rougeL=rougeL,
            bertscore_f1=bertscore_f1,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            correctness=correctness,
        )

    def test_avg_rouge1_averaged(self):
        results = [
            self._make_result_with_scores("q1", 0, rouge1=0.4),
            self._make_result_with_scores("q2", 0, rouge1=0.8),
        ]
        m = compute_generation_metrics(results, "stpo", "llama")
        assert m.avg_rouge1 == pytest.approx(0.6)

    def test_avg_bertscore_averaged(self):
        results = [
            self._make_result_with_scores("q1", 0, bertscore_f1=0.7),
            self._make_result_with_scores("q2", 0, bertscore_f1=0.9),
        ]
        m = compute_generation_metrics(results, "stpo", "llama")
        assert m.avg_bertscore_f1 == pytest.approx(0.8)

    def test_avg_judge_scores_averaged(self):
        results = [
            self._make_result_with_scores("q1", 0, faithfulness=0.6, answer_relevance=0.7, correctness=0.8),
            self._make_result_with_scores("q2", 0, faithfulness=1.0, answer_relevance=0.9, correctness=0.6),
        ]
        m = compute_generation_metrics(results, "stpo", "llama")
        assert m.avg_faithfulness == pytest.approx(0.8)
        assert m.avg_answer_relevance == pytest.approx(0.8)
        assert m.avg_correctness == pytest.approx(0.7)

    def test_empty_results_quality_zeros(self):
        m = compute_generation_metrics([], "stpo", "llama")
        assert m.avg_rouge1 == 0.0
        assert m.avg_bertscore_f1 == 0.0
        assert m.avg_faithfulness == 0.0
