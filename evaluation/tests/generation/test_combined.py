"""Tests for the combined knowledge base generation evaluation module.

Uses a stub Generator implementation, monkeypatched _score_with_ragas,
and temporary JSON files to test the combined-KB generation pipeline
without requiring Ollama, RAGAS, or real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation.generation.combined import (
    run_and_report_combined,
    run_combined_generation_evaluation,
)
from evaluation.generation.metrics import GenerationEvalResult
from evaluation.tests.conftest import fake_ragas_scores
from tests.conftest import StubGenerator


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    """Write a minimal chunk JSON file."""
    data = {"chunks": chunks}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_eval(path: Path, questions: list[dict]) -> None:
    """Write a minimal evaluation JSON file."""
    data = {
        "metadata": {"version": "1.0"},
        "questions": questions,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _chunk(cid: str, text: str = "") -> dict:
    """Create a minimal chunk dict."""
    return {"chunk_id": cid, "text": text or f"text-{cid}", "metadata": {}}


def _question(qid: str, relevant: list[str], **kwargs) -> dict:
    """Create a minimal question dict."""
    q = {
        "id": qid,
        "question": f"Question {qid}?",
        "reference_answer": f"Answer for {qid}.",
        "category": "direct",
        "relevant_chunks": relevant,
        "expected_abstention": False,
    }
    q.update(kwargs)
    return q


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_kb_setup(tmp_path):
    """Set up two KBs with distinct chunks and shared questions."""
    # KB1: stpo -- 3 chunks
    cp1 = tmp_path / "stpo-chunks.json"
    _write_chunks(cp1, [
        _chunk("stpo-1", "Study period is 4 semesters."),
        _chunk("stpo-2", "Thesis has 30 credits."),
        _chunk("stpo-3", "Examination rules apply."),
    ])
    ep1 = tmp_path / "eval-stpo.json"
    _write_eval(ep1, [
        _question("q1", ["stpo-1"]),
        _question("q2", ["stpo-2"]),
        _question("q3", []),
    ])

    # KB2: faq -- 3 chunks
    cp2 = tmp_path / "faq-chunks.json"
    _write_chunks(cp2, [
        _chunk("faq-1", "The program lasts 4 semesters."),
        _chunk("faq-2", "You need 120 ECTS total."),
        _chunk("faq-3", "Study abroad info."),
    ])
    ep2 = tmp_path / "eval-faq.json"
    _write_eval(ep2, [
        _question("q1", ["faq-1"]),
        _question("q2", []),
        _question("q3", ["faq-2"]),
    ])

    return {
        "chunk_paths": {"stpo": cp1, "faq": cp2},
        "eval_paths": {"stpo": ep1, "faq": ep2},
    }


# ---------------------------------------------------------------------------
# Tests: run_combined_generation_evaluation
# ---------------------------------------------------------------------------


@patch("evaluation.generation.evaluate._score_with_ragas", fake_ragas_scores)
class TestRunCombinedGenerationEvaluation:
    def test_result_count(self, two_kb_setup, stub_judge):
        """3 answerable questions x 2 distractor levels = 6 results."""
        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0, 1],
            **two_kb_setup,
        )
        assert len(results) == 6

    def test_merges_relevant_chunks_from_both_kbs(self, two_kb_setup, stub_judge):
        """Question q1 has relevant chunks in both KBs (stpo-1, faq-1)."""
        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        q1_results = [r for r in results if r.question_id == "q1"]
        assert len(q1_results) == 1
        assert set(q1_results[0].context_chunk_ids) == {"stpo-1", "faq-1"}

    def test_question_with_single_kb_relevance(self, two_kb_setup, stub_judge):
        """Question q2 only has relevant chunks in stpo (stpo-2)."""
        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        q2_results = [r for r in results if r.question_id == "q2"]
        assert len(q2_results) == 1
        assert "stpo-2" in q2_results[0].context_chunk_ids

    def test_distractors_from_merged_corpus(self, tmp_path, stub_judge):
        """With distractors, context includes non-relevant chunks."""
        cp1 = tmp_path / "kb1.json"
        _write_chunks(cp1, [
            _chunk("r1", "The study period is four semesters."),
            _chunk("d1", "The study program includes a seminar."),
            _chunk("d2", "The study regulations cover examinations."),
        ])
        cp2 = tmp_path / "kb2.json"
        _write_chunks(cp2, [
            _chunk("d3", "Students study a range of modules."),
        ])
        ep1 = tmp_path / "eval1.json"
        _write_eval(ep1, [_question("q1", ["r1"],
                          question="How long is the study period?")])
        ep2 = tmp_path / "eval2.json"
        _write_eval(ep2, [_question("q1", [])])

        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            chunk_paths={"kb1": cp1, "kb2": cp2},
            eval_paths={"kb1": ep1, "kb2": ep2},
            judge=stub_judge,
            distractor_levels=[2],
        )
        chunk_ids = set(results[0].context_chunk_ids)
        assert "r1" in chunk_ids
        assert len(chunk_ids) == 3

    def test_all_distractor_levels(self, two_kb_setup, stub_judge):
        """Default distractor levels 0-10 produce 11 results per question."""
        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            **two_kb_setup,
        )
        # 3 answerable questions x 11 levels = 33
        assert len(results) == 33

    def test_skips_unanswerable(self, tmp_path, stub_judge):
        """Questions with expected_abstention=True are skipped."""
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("c1")])
        _write_eval(ep, [
            _question("q1", [], expected_abstention=True,
                      category="unanswerable"),
        ])
        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            judge=stub_judge,
            distractor_levels=[0],
        )
        assert results == []

    def test_progress_callback(self, two_kb_setup, stub_judge):
        """Progress callback is invoked for each question x level pair."""
        calls = []

        def callback(qid, n_dist):
            calls.append((qid, n_dist))

        run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0, 1],
            progress_callback=callback,
            **two_kb_setup,
        )
        assert len(calls) == 6

    def test_returns_generation_eval_results(self, two_kb_setup, stub_judge):
        """All returned objects are GenerationEvalResult instances."""
        results = run_combined_generation_evaluation(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        for r in results:
            assert isinstance(r, GenerationEvalResult)
            assert r.generated_answer == "stub answer"
            assert r.reference_answer != ""


# ---------------------------------------------------------------------------
# Tests: run_and_report_combined
# ---------------------------------------------------------------------------


@patch("evaluation.generation.evaluate._score_with_ragas", fake_ragas_scores)
class TestRunAndReportCombined:
    def test_report_structure(self, two_kb_setup, stub_judge):
        """Report contains all expected top-level keys."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        assert "combination" in report
        assert "eval_files" in report
        assert "config" in report
        assert "metrics" in report
        assert "results" in report

    def test_combination_from_kb_names(self, two_kb_setup, stub_judge):
        """Default combination name is KB names joined with +."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        assert report["combination"] == "faq+stpo"

    def test_custom_combination_name(self, two_kb_setup, stub_judge):
        """Custom combination name overrides default."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            combination_name="stpo+faq-custom",
            **two_kb_setup,
        )
        assert report["combination"] == "stpo+faq-custom"
        assert report["config"]["combination"] == "stpo+faq-custom"

    def test_config_fields(self, two_kb_setup, stub_judge):
        """Config contains all expected fields."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0, 1, 2],
            **two_kb_setup,
        )
        config = report["config"]
        assert config["distractor_levels"] == [0, 1, 2]
        assert config["generator_model"] == "stub-model"
        assert config["corpus_size"] == 6  # 3 + 3 chunks
        assert sorted(config["knowledge_bases"]) == ["faq", "stpo"]
        assert config["judge_batch_size"] == stub_judge.batch_size

    def test_metrics_fields(self, two_kb_setup, stub_judge):
        """Metrics contain expected aggregation fields."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        metrics = report["metrics"]
        assert metrics["num_results"] == 3  # 3 answerable questions x 1 level
        assert metrics["num_queries"] == 3
        assert metrics["knowledge_base"] == "faq+stpo"
        assert metrics["model"] == "stub-model"

    def test_results_serialised(self, two_kb_setup, stub_judge):
        """Results are serialised as plain dicts (not dataclasses)."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        for r in report["results"]:
            assert isinstance(r, dict)
            assert "question_id" in r
            assert "generated_answer" in r
            assert "reference_answer" in r
            assert "num_distractors" in r
            assert "context_chunk_ids" in r

    def test_ragas_scores_in_report(self, two_kb_setup, stub_judge):
        """RAGAS scores are included in serialised results."""
        report = run_and_report_combined(
            generator=StubGenerator(answer="stub answer"),
            judge=stub_judge,
            distractor_levels=[0],
            **two_kb_setup,
        )
        for r in report["results"]:
            assert "faithfulness" in r
            assert "answer_relevance" in r
            assert "correctness" in r
