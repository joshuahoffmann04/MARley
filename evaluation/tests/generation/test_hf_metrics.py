"""Tests for the HuggingFace evaluate metrics helpers (ROUGE + BERTScore).

ROUGE is tested with real computation — it is fast and fully deterministic.
BERTScore tests are marked as integration because they require a model
download on first run.
"""

from __future__ import annotations

import pytest

from evaluation.generation.hf_metrics import compute_bertscore, compute_rouge


# ---------------------------------------------------------------------------
# TestComputeRouge
# ---------------------------------------------------------------------------


class TestComputeRouge:
    def test_empty_returns_empty(self):
        assert compute_rouge([], []) == []

    def test_returns_one_dict_per_pair(self):
        preds = ["The study period is 4 semesters.", "30 credits for the thesis."]
        refs = ["4 semesters.", "30 credits."]
        results = compute_rouge(preds, refs)
        assert len(results) == 2

    def test_result_dict_has_three_keys(self):
        results = compute_rouge(["hello world"], ["hello world"])
        assert set(results[0].keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_identical_strings_score_one(self):
        text = "The standard study period is 4 semesters."
        results = compute_rouge([text], [text])
        assert results[0]["rouge1"] == pytest.approx(1.0)
        assert results[0]["rougeL"] == pytest.approx(1.0)

    def test_unrelated_strings_score_near_zero(self):
        results = compute_rouge(["banana apple mango"], ["car road bridge"])
        assert results[0]["rouge1"] == pytest.approx(0.0)

    def test_scores_in_range(self):
        preds = ["The study period is 4 semesters.", "Thesis worth 30 credits."]
        refs = ["4 semesters standard period.", "30 credits master thesis."]
        results = compute_rouge(preds, refs)
        for r in results:
            assert 0.0 <= r["rouge1"] <= 1.0
            assert 0.0 <= r["rouge2"] <= 1.0
            assert 0.0 <= r["rougeL"] <= 1.0

    def test_partial_overlap(self):
        results = compute_rouge(
            ["The study period is 4 semesters."],
            ["The study period is 2 semesters."],
        )
        # Partially overlapping — between 0 and 1
        assert 0.0 < results[0]["rouge1"] < 1.0


# ---------------------------------------------------------------------------
# TestComputeBertscore (integration — requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestComputeBertscore:
    def test_empty_returns_empty(self):
        assert compute_bertscore([], []) == []

    def test_returns_one_float_per_pair(self):
        preds = ["The study period is 4 semesters.", "30 credits."]
        refs = ["4 semesters.", "30 credits for the thesis."]
        results = compute_bertscore(preds, refs)
        assert len(results) == 2
        assert all(isinstance(v, float) for v in results)

    def test_identical_strings_score_high(self):
        text = "The standard study period is 4 semesters."
        results = compute_bertscore([text], [text])
        assert results[0] >= 0.99

    def test_scores_in_range(self):
        preds = ["The study period is 4 semesters."]
        refs = ["4 semesters standard period."]
        results = compute_bertscore(preds, refs)
        assert 0.0 <= results[0] <= 1.0
