"""Tests for evaluation item preparation from generation results.

Covers prepare_generation_items() and prepare_items_from_results()
with sample generation result data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.manual.models import EvaluationItem
from evaluation.manual.prepare import (
    prepare_generation_items,
    prepare_items_from_results,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


SAMPLE_GENERATION_RESULTS = {
    "eval_file": "data/testing/evaluation-stpo.json",
    "config": {
        "distractor_levels": [0, 1, 2],
        "generator_model": "llama3.1:latest",
        "corpus_size": 151,
        "knowledge_base": "stpo",
    },
    "metrics": {},
    "results": [
        {
            "question_id": "eval-001",
            "num_distractors": 0,
            "generated_answer": "4 semesters.",
            "reference_answer": "The study period is 4 semesters.",
            "context_chunk_ids": ["par-7-txt-1"],
        },
        {
            "question_id": "eval-001",
            "num_distractors": 1,
            "generated_answer": "4 semesters for the master's.",
            "reference_answer": "The study period is 4 semesters.",
            "context_chunk_ids": ["par-7-txt-1", "par-3-txt-1"],
        },
        {
            "question_id": "eval-002",
            "num_distractors": 0,
            "generated_answer": "Master of Science.",
            "reference_answer": "Graduates receive M.Sc.",
            "context_chunk_ids": ["par-3-txt-1"],
        },
    ],
}


SAMPLE_EVAL_DATASET = {
    "metadata": {"knowledge_base": "stpo"},
    "questions": [
        {
            "id": "eval-001",
            "question": "How long is the study period?",
            "reference_answer": "The study period is 4 semesters.",
            "category": "direct",
            "relevant_chunks": ["par-7-txt-1"],
            "expected_abstention": False,
        },
        {
            "id": "eval-002",
            "question": "What degree do graduates receive?",
            "reference_answer": "Graduates receive M.Sc.",
            "category": "direct",
            "relevant_chunks": ["par-3-txt-1"],
            "expected_abstention": False,
        },
        {
            "id": "eval-076",
            "question": "What is the tuition fee?",
            "reference_answer": "",
            "category": "unanswerable",
            "relevant_chunks": [],
            "expected_abstention": True,
        },
    ],
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# TestPrepareGenerationItems
# ---------------------------------------------------------------------------


class TestPrepareGenerationItems:
    """Tests for prepare_generation_items()."""

    def test_correct_item_count(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)

        items = prepare_generation_items(result_path, "stpo")
        assert len(items) == 3

    def test_item_id_format(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)

        items = prepare_generation_items(result_path, "stpo")
        assert items[0].id == "gen-stpo-eval-001-d0"
        assert items[1].id == "gen-stpo-eval-001-d1"
        assert items[2].id == "gen-stpo-eval-002-d0"

    def test_metadata_fields(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)

        items = prepare_generation_items(result_path, "stpo")
        meta = items[0].metadata
        assert meta["question_id"] == "eval-001"
        assert meta["knowledge_base"] == "stpo"
        assert meta["num_distractors"] == 0
        assert meta["evaluation_type"] == "generation"
        assert meta["generator_model"] == "llama3.1:latest"
        assert meta["context_chunk_ids"] == ["par-7-txt-1"]

    def test_generated_and_reference_answer(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)

        items = prepare_generation_items(result_path, "stpo")
        assert items[0].generated_answer == "4 semesters."
        assert items[0].reference_answer == "The study period is 4 semesters."

    def test_enriches_from_eval_dataset(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        dataset_path = tmp_path / "eval-dataset.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)
        _write_json(dataset_path, SAMPLE_EVAL_DATASET)

        items = prepare_generation_items(result_path, "stpo", eval_dataset_path=dataset_path)
        assert items[0].question == "How long is the study period?"
        assert items[0].category == "direct"
        assert items[0].expected_abstention is False

    def test_without_eval_dataset_defaults(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)

        items = prepare_generation_items(result_path, "stpo")
        assert items[0].question == ""
        assert items[0].category == ""
        assert items[0].expected_abstention is False

    def test_all_items_are_evaluation_items(self, tmp_path: Path):
        result_path = tmp_path / "gen-eval.json"
        _write_json(result_path, SAMPLE_GENERATION_RESULTS)

        items = prepare_generation_items(result_path, "stpo")
        assert all(isinstance(i, EvaluationItem) for i in items)


# ---------------------------------------------------------------------------
# TestPrepareItemsFromResults
# ---------------------------------------------------------------------------


class TestPrepareItemsFromResults:
    """Tests for prepare_items_from_results()."""

    def test_correct_count(self):
        results = SAMPLE_GENERATION_RESULTS["results"]
        items = prepare_items_from_results(results, "stpo", "llama3.1:latest")
        assert len(items) == 3

    def test_with_question_metadata(self):
        results = SAMPLE_GENERATION_RESULTS["results"]
        meta = {
            "eval-001": {"question": "Q1?", "category": "direct", "expected_abstention": False},
            "eval-002": {"question": "Q2?", "category": "multi-source", "expected_abstention": False},
        }
        items = prepare_items_from_results(results, "stpo", "llama3.1:latest", question_metadata=meta)
        assert items[0].question == "Q1?"
        assert items[0].category == "direct"
        assert items[2].category == "multi-source"

    def test_without_metadata_defaults(self):
        results = [SAMPLE_GENERATION_RESULTS["results"][0]]
        items = prepare_items_from_results(results, "faq-ao", "mistral")
        assert items[0].metadata["generator_model"] == "mistral"
        assert items[0].metadata["knowledge_base"] == "faq-ao"
        assert items[0].category == ""
