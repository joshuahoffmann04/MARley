"""Tests for the manual evaluation data model.

Covers the Judgement enum, EvaluationItem and ManualJudgement data classes,
and all I/O functions (save/load items, save/load judgements).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.manual.models import (
    ABSTENTION_JUDGEMENTS,
    ANSWER_JUDGEMENTS,
    EvaluationItem,
    Judgement,
    ManualJudgement,
    load_items,
    load_judgements,
    save_items,
    save_judgement,
)


# ---------------------------------------------------------------------------
# TestJudgement
# ---------------------------------------------------------------------------


class TestJudgement:
    """Tests for the Judgement enum."""

    def test_all_six_values_exist(self):
        assert len(Judgement) == 6

    def test_string_values(self):
        assert Judgement.CORRECT.value == "correct"
        assert Judgement.PARTIALLY_CORRECT.value == "partially_correct"
        assert Judgement.INCORRECT.value == "incorrect"
        assert Judgement.CORRECT_ABSTENTION.value == "correct_abstention"
        assert Judgement.INCORRECT_ABSTENTION.value == "incorrect_abstention"
        assert Judgement.MISSING_ABSTENTION.value == "missing_abstention"

    def test_is_str_enum(self):
        assert isinstance(Judgement.CORRECT, str)
        assert Judgement.CORRECT == "correct"

    def test_answer_judgements_group(self):
        assert ANSWER_JUDGEMENTS == frozenset({
            Judgement.CORRECT,
            Judgement.PARTIALLY_CORRECT,
            Judgement.INCORRECT,
        })

    def test_abstention_judgements_group(self):
        assert ABSTENTION_JUDGEMENTS == frozenset({
            Judgement.CORRECT_ABSTENTION,
            Judgement.INCORRECT_ABSTENTION,
            Judgement.MISSING_ABSTENTION,
        })

    def test_groups_are_disjoint(self):
        assert ANSWER_JUDGEMENTS & ABSTENTION_JUDGEMENTS == frozenset()

    def test_groups_cover_all(self):
        assert ANSWER_JUDGEMENTS | ABSTENTION_JUDGEMENTS == frozenset(Judgement)

    def test_construct_from_string(self):
        assert Judgement("correct") == Judgement.CORRECT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Judgement("invalid")


# ---------------------------------------------------------------------------
# TestEvaluationItem
# ---------------------------------------------------------------------------


class TestEvaluationItem:
    """Tests for the EvaluationItem data class."""

    def test_create_with_all_fields(self):
        item = EvaluationItem(
            id="gen-stpo-eval-001-d0",
            question="How long is the study period?",
            generated_answer="4 semesters.",
            reference_answer="4 semesters.",
            category="direct",
            expected_abstention=False,
            metadata={"knowledge_base": "stpo"},
        )
        assert item.id == "gen-stpo-eval-001-d0"
        assert item.expected_abstention is False

    def test_default_metadata(self):
        item = EvaluationItem(
            id="test", question="q", generated_answer="a",
            reference_answer="r", category="direct", expected_abstention=False,
        )
        assert item.metadata == {}


# ---------------------------------------------------------------------------
# TestManualJudgement
# ---------------------------------------------------------------------------


class TestManualJudgement:
    """Tests for the ManualJudgement data class."""

    def test_create_with_enum(self):
        j = ManualJudgement(
            item_id="gen-stpo-eval-001-d0",
            judgement=Judgement.CORRECT,
            notes="Looks good.",
        )
        assert j.judgement == Judgement.CORRECT
        assert j.notes == "Looks good."

    def test_auto_timestamp(self):
        j = ManualJudgement(item_id="test", judgement=Judgement.INCORRECT)
        assert j.timestamp != ""
        assert "T" in j.timestamp  # ISO format

    def test_string_coercion(self):
        j = ManualJudgement(item_id="test", judgement="partially_correct")
        assert j.judgement == Judgement.PARTIALLY_CORRECT

    def test_preserves_explicit_timestamp(self):
        j = ManualJudgement(
            item_id="test", judgement=Judgement.CORRECT,
            timestamp="2026-03-10T12:00:00+00:00",
        )
        assert j.timestamp == "2026-03-10T12:00:00+00:00"


# ---------------------------------------------------------------------------
# TestSaveLoadItems
# ---------------------------------------------------------------------------


class TestSaveLoadItems:
    """Tests for save_items() and load_items()."""

    def test_round_trip(self, tmp_path: Path):
        items = [
            EvaluationItem(
                id="item-1", question="Q1", generated_answer="A1",
                reference_answer="R1", category="direct",
                expected_abstention=False, metadata={"kb": "stpo"},
            ),
            EvaluationItem(
                id="item-2", question="Q2", generated_answer="A2",
                reference_answer="R2", category="unanswerable",
                expected_abstention=True, metadata={"kb": "stpo"},
            ),
        ]

        path = tmp_path / "items.json"
        save_items(items, path, metadata={"source": "test"})
        loaded = load_items(path)

        assert len(loaded) == 2
        assert loaded[0].id == "item-1"
        assert loaded[1].expected_abstention is True
        assert loaded[0].metadata == {"kb": "stpo"}

    def test_file_contains_metadata(self, tmp_path: Path):
        items = [
            EvaluationItem(
                id="item-1", question="Q", generated_answer="A",
                reference_answer="R", category="direct",
                expected_abstention=False,
            ),
        ]
        path = tmp_path / "items.json"
        save_items(items, path, metadata={"source": "test", "total_items": 1})

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["metadata"]["source"] == "test"
        assert raw["metadata"]["total_items"] == 1

    def test_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "subdir" / "items.json"
        save_items([], path)
        assert path.exists()


# ---------------------------------------------------------------------------
# TestSaveLoadJudgements
# ---------------------------------------------------------------------------


class TestSaveLoadJudgements:
    """Tests for save_judgement() and load_judgements()."""

    def test_save_creates_file(self, tmp_path: Path):
        path = tmp_path / "judgements.json"
        j = ManualJudgement(item_id="item-1", judgement=Judgement.CORRECT)
        save_judgement(j, path)
        assert path.exists()

    def test_save_appends(self, tmp_path: Path):
        path = tmp_path / "judgements.json"
        save_judgement(
            ManualJudgement(item_id="item-1", judgement=Judgement.CORRECT), path,
        )
        save_judgement(
            ManualJudgement(item_id="item-2", judgement=Judgement.INCORRECT), path,
        )

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert len(raw["judgements"]) == 2

    def test_load_deduplicates_by_latest(self, tmp_path: Path):
        path = tmp_path / "judgements.json"
        save_judgement(
            ManualJudgement(
                item_id="item-1", judgement=Judgement.INCORRECT,
                timestamp="2026-03-10T12:00:00",
            ),
            path,
        )
        save_judgement(
            ManualJudgement(
                item_id="item-1", judgement=Judgement.CORRECT,
                timestamp="2026-03-10T13:00:00",
            ),
            path,
        )

        loaded = load_judgements(path)
        assert len(loaded) == 1
        assert loaded[0].judgement == Judgement.CORRECT

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        loaded = load_judgements(path)
        assert loaded == []

    def test_metadata_updated(self, tmp_path: Path):
        path = tmp_path / "judgements.json"
        save_judgement(
            ManualJudgement(item_id="item-1", judgement=Judgement.CORRECT), path,
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "started" in raw["metadata"]
        assert "last_updated" in raw["metadata"]
        assert raw["metadata"]["last_updated"] != ""
