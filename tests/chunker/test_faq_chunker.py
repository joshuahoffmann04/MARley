"""Tests for the FAQ chunker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tiktoken

from src.marley.models import QualityFlag
from src.marley.chunker.faq_chunker import (
    FAQChunk,
    FAQChunkingResult,
    FAQChunkingStats,
    FAQChunkMetadata,
    FAQDataset,
    FAQEntry,
    _build_chunk_id,
    _compute_stats,
    _format_chunk_text,
    _validate_entry,
    chunk_faq,
    load,
    save,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FAQ_STPO_PATH = PROJECT_ROOT / "data" / "knowledgebase" / "faq-stpo.json"
FAQ_AO_PATH = PROJECT_ROOT / "data" / "knowledgebase" / "faq-ao.json"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestFormatChunkText:
    def test_normal_qa(self):
        text = _format_chunk_text("What is X?", "X is Y.")
        assert text == "Question: What is X?\nAnswer: X is Y."

    def test_strips_whitespace(self):
        text = _format_chunk_text("  What?  ", "  Answer.  ")
        assert text == "Question: What?\nAnswer: Answer."

    def test_multiline_answer(self):
        text = _format_chunk_text("Q?", "Line 1.\nLine 2.")
        assert "Question: Q?" in text
        assert "Answer: Line 1.\nLine 2." in text


class TestBuildChunkId:
    def test_stpo_format(self):
        assert _build_chunk_id("faq-stpo", "stpo-0001") == "faq-stpo-stpo-0001"

    def test_ao_format(self):
        assert _build_chunk_id("faq-ao", "ao-0001") == "faq-ao-ao-0001"

    def test_custom_id(self):
        assert _build_chunk_id("faq-stpo", "custom-id") == "faq-stpo-custom-id"


class TestValidateEntry:
    def _make_entry(self, **overrides):
        defaults = {"id": "stpo-0001", "question": "Q?", "answer": "A.", "source": "§1"}
        defaults.update(overrides)
        return FAQEntry(**defaults)

    def test_valid_entry(self):
        flags: list[QualityFlag] = []
        result = _validate_entry(self._make_entry(), 0, set(), flags)
        assert result is True
        assert not flags

    def test_missing_id(self):
        flags: list[QualityFlag] = []
        result = _validate_entry(self._make_entry(id=""), 0, set(), flags)
        assert result is False
        assert flags[0].code == "FAQ_ENTRY_INVALID"

    def test_missing_question(self):
        flags: list[QualityFlag] = []
        result = _validate_entry(self._make_entry(question=""), 0, set(), flags)
        assert result is False
        assert flags[0].code == "FAQ_EMPTY_QUESTION"

    def test_missing_answer(self):
        flags: list[QualityFlag] = []
        result = _validate_entry(self._make_entry(answer=""), 0, set(), flags)
        assert result is False
        assert flags[0].code == "FAQ_EMPTY_ANSWER"

    def test_duplicate_id(self):
        flags: list[QualityFlag] = []
        seen = {"stpo-0001"}
        result = _validate_entry(self._make_entry(), 0, seen, flags)
        assert result is False
        assert flags[0].code == "FAQ_ID_DUPLICATE"


class TestLoad:
    def test_valid_json(self, tmp_path):
        data = {
            "metadata": {"source": "faq-test"},
            "entries": [
                {"id": "t-001", "question": "Q?", "answer": "A.", "source": "§1"},
            ],
        }
        path = tmp_path / "faq.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        dataset = load(path)
        assert dataset.faq_source == "faq-test"
        assert len(dataset.entries) == 1
        assert dataset.entries[0].id == "t-001"

    def test_missing_entries_key(self, tmp_path):
        path = tmp_path / "faq.json"
        path.write_text(json.dumps({"metadata": {"source": "test"}}), encoding="utf-8")
        dataset = load(path)
        assert len(dataset.entries) == 0

    def test_empty_entries(self, tmp_path):
        data = {"metadata": {"source": "test"}, "entries": []}
        path = tmp_path / "faq.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        dataset = load(path)
        assert len(dataset.entries) == 0

    def test_non_dict_entry_skipped(self, tmp_path):
        data = {"metadata": {"source": "test"}, "entries": ["not a dict", 42]}
        path = tmp_path / "faq.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        dataset = load(path)
        assert len(dataset.entries) == 0


class TestChunkFAQ:
    def _make_dataset(self, entries):
        return FAQDataset(
            faq_source="faq-test",
            entries=[FAQEntry(**e) for e in entries],
        )

    def test_single_entry(self):
        ds = self._make_dataset([
            {"id": "t-001", "question": "Q?", "answer": "A.", "source": "§1"},
        ])
        result = chunk_faq(ds)
        assert result.stats.total_chunks == 1
        assert result.chunks[0].chunk_id == "faq-test-t-001"
        assert result.chunks[0].text.startswith("Question:")

    def test_multiple_entries(self):
        ds = self._make_dataset([
            {"id": "t-001", "question": "Q1?", "answer": "A1.", "source": "§1"},
            {"id": "t-002", "question": "Q2?", "answer": "A2.", "source": "§2"},
        ])
        result = chunk_faq(ds)
        assert result.stats.total_chunks == 2

    def test_empty_dataset(self):
        ds = self._make_dataset([])
        result = chunk_faq(ds)
        assert result.stats.total_chunks == 0
        assert not any(f.code == "FAQ_ALL_SKIPPED" for f in result.quality_flags)

    def test_mixed_valid_invalid(self):
        ds = self._make_dataset([
            {"id": "t-001", "question": "Q?", "answer": "A.", "source": "§1"},
            {"id": "", "question": "Q?", "answer": "A.", "source": "§2"},
            {"id": "t-003", "question": "Q?", "answer": "A.", "source": "§3"},
        ])
        result = chunk_faq(ds)
        assert result.stats.total_chunks == 2
        assert result.stats.entries_skipped == 1


class TestComputeStats:
    def test_normal_stats(self):
        encoder = tiktoken.get_encoding("cl100k_base")
        chunks = [
            FAQChunk(
                chunk_id="c1", chunk_type="faq", text="short",
                token_count=5,
                metadata=FAQChunkMetadata("s", "", "id", "§1", 0),
            ),
            FAQChunk(
                chunk_id="c2", chunk_type="faq", text="longer text here",
                token_count=15,
                metadata=FAQChunkMetadata("s", "", "id2", "§2", 0),
            ),
        ]
        stats = _compute_stats(chunks, entries_total=2, entries_skipped=0)
        assert stats.total_chunks == 2
        assert stats.min_tokens == 5
        assert stats.max_tokens == 15
        assert stats.total_tokens == 20

    def test_empty_chunks(self):
        stats = _compute_stats([], entries_total=3, entries_skipped=3)
        assert stats.total_chunks == 0
        assert stats.entries_total == 3
        assert stats.entries_skipped == 3


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# --- FAQ-StPO fixtures and tests ---

stpo_skip = pytest.mark.skipif(
    not FAQ_STPO_PATH.exists(), reason="faq-stpo.json not found"
)


@pytest.fixture(scope="module")
def stpo_dataset():
    return load(FAQ_STPO_PATH)


@pytest.fixture(scope="module")
def stpo_result(stpo_dataset):
    return chunk_faq(stpo_dataset, source_file=str(FAQ_STPO_PATH))


@stpo_skip
class TestFAQStPOChunking:
    def test_total_chunks(self, stpo_result):
        assert stpo_result.stats.total_chunks == 999

    def test_all_chunks_have_text(self, stpo_result):
        assert all(c.text.strip() for c in stpo_result.chunks)

    def test_all_chunks_have_metadata(self, stpo_result):
        for c in stpo_result.chunks:
            assert c.metadata.faq_source == "faq-stpo"
            assert c.metadata.faq_id

    def test_chunk_ids_unique(self, stpo_result):
        ids = [c.chunk_id for c in stpo_result.chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_start_with_faq_stpo(self, stpo_result):
        assert all(c.chunk_id.startswith("faq-stpo-") for c in stpo_result.chunks)


# --- FAQ-AO fixtures and tests ---

ao_skip = pytest.mark.skipif(
    not FAQ_AO_PATH.exists(), reason="faq-ao.json not found"
)


@pytest.fixture(scope="module")
def ao_dataset():
    return load(FAQ_AO_PATH)


@pytest.fixture(scope="module")
def ao_result(ao_dataset):
    return chunk_faq(ao_dataset, source_file=str(FAQ_AO_PATH))


@ao_skip
class TestFAQAOChunking:
    def test_total_chunks(self, ao_result):
        assert ao_result.stats.total_chunks == 50

    def test_all_chunks_have_text(self, ao_result):
        assert all(c.text.strip() for c in ao_result.chunks)

    def test_chunk_ids_unique(self, ao_result):
        ids = [c.chunk_id for c in ao_result.chunks]
        assert len(ids) == len(set(ids))


# --- Content and quality tests (use StPO) ---

@stpo_skip
class TestChunkContent:
    def test_text_starts_with_question(self, stpo_result):
        for c in stpo_result.chunks:
            assert c.text.startswith("Question:")

    def test_text_contains_answer(self, stpo_result):
        for c in stpo_result.chunks:
            assert "\nAnswer:" in c.text

    def test_source_reference_populated(self, stpo_result):
        for c in stpo_result.chunks:
            assert c.metadata.source_reference


@stpo_skip
class TestQualityFlags:
    def test_no_error_flags(self, stpo_result):
        errors = [f for f in stpo_result.quality_flags if f.severity == "error"]
        assert not errors

    def test_stats_match_chunks(self, stpo_result):
        assert stpo_result.stats.entries_processed == len(stpo_result.chunks)


@stpo_skip
class TestSaveAndLoad:
    def test_json_roundtrip(self, stpo_result, tmp_path):
        out = tmp_path / "faq-chunks.json"
        save(stpo_result, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["stats"]["total_chunks"] == stpo_result.stats.total_chunks
        assert len(data["chunks"]) == len(stpo_result.chunks)

    def test_creates_parent_dirs(self, stpo_result, tmp_path):
        out = tmp_path / "sub" / "dir" / "faq.json"
        result_path = save(stpo_result, out)
        assert result_path.exists()
