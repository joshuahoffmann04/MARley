"""Tests for shared data classes and utilities in src.marley.models."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from src.marley.models.abstention import AbstentionResult
from src.marley.models.chunking import compute_token_stats
from src.marley.models.extraction import ExtractionResult, Section, Table
from src.marley.models.generation import GenerationResult
from src.marley.models.io import save_json
from src.marley.models.quality import QualityFlag
from src.marley.models.retrieval import RetrievalResult, load_chunks, validate_corpus


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_construction(self) -> None:
        r = RetrievalResult(chunk_id="c1", text="hello", score=0.9, metadata={"k": "v"})
        assert r.chunk_id == "c1"
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.metadata == {"k": "v"}

    def test_asdict(self) -> None:
        r = RetrievalResult(chunk_id="c1", text="t", score=0.5, metadata={})
        d = asdict(r)
        assert d == {"chunk_id": "c1", "text": "t", "score": 0.5, "metadata": {}}


# ---------------------------------------------------------------------------
# load_chunks
# ---------------------------------------------------------------------------


class TestLoadChunks:
    """Tests for the load_chunks utility."""

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        data = {"chunks": [
            {"chunk_id": "c1", "text": "hello", "metadata": {}},
        ]}
        f = tmp_path / "chunks.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_chunks(f)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_chunks("nonexistent/file.json")

    def test_missing_chunks_key_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"data": []}), encoding="utf-8")
        with pytest.raises(KeyError):
            load_chunks(f)

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        data = {"chunks": [{"chunk_id": "c1", "text": "t", "metadata": {}}]}
        f = tmp_path / "chunks.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = load_chunks(Path(f))
        assert len(result) == 1

    def test_empty_chunks_list(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"chunks": []}), encoding="utf-8")
        assert load_chunks(f) == []


# ---------------------------------------------------------------------------
# validate_corpus
# ---------------------------------------------------------------------------


class TestValidateCorpus:
    """Tests for the validate_corpus utility."""

    def test_valid_corpus_passes(self) -> None:
        corpus = [{"chunk_id": "c1", "text": "t", "metadata": {}}]
        validate_corpus(corpus)  # should not raise

    def test_empty_corpus_passes(self) -> None:
        validate_corpus([])  # should not raise

    def test_missing_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id"):
            validate_corpus([{"text": "t", "metadata": {}}])

    def test_missing_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            validate_corpus([{"chunk_id": "c1", "metadata": {}}])

    def test_missing_metadata_raises(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            validate_corpus([{"chunk_id": "c1", "text": "t"}])

    def test_multiple_missing_keys_reported(self) -> None:
        with pytest.raises(ValueError, match="missing required keys"):
            validate_corpus([{}])

    def test_error_reports_index(self) -> None:
        corpus = [
            {"chunk_id": "c1", "text": "t", "metadata": {}},
            {"chunk_id": "c2", "text": "t"},  # missing metadata
        ]
        with pytest.raises(ValueError, match="index 1"):
            validate_corpus(corpus)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class TestTable:
    """Tests for the Table dataclass."""

    def test_construction(self) -> None:
        t = Table(table_id="t1", page=3, headers=["A", "B"], rows=[["1", "2"]])
        assert t.table_id == "t1"
        assert t.page == 3
        assert t.headers == ["A", "B"]
        assert t.rows == [["1", "2"]]

    def test_asdict(self) -> None:
        t = Table(table_id="t1", page=1, headers=[], rows=[])
        d = asdict(t)
        assert d == {"table_id": "t1", "page": 1, "headers": [], "rows": []}


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


class TestSection:
    """Tests for the Section dataclass."""

    def test_construction_with_defaults(self) -> None:
        s = Section(
            section_id="s1", label="§ 1", title="Scope",
            kind="paragraph", start_page=1, end_page=2, text="Some text.",
        )
        assert s.section_id == "s1"
        assert s.tables == []
        assert s.parent_section_id is None

    def test_tables_default_factory(self) -> None:
        s1 = Section("a", "", "", "part", 1, 1, "")
        s2 = Section("b", "", "", "part", 1, 1, "")
        assert s1.tables is not s2.tables

    def test_parent_section_id(self) -> None:
        s = Section("s2", "", "", "paragraph", 1, 1, "", parent_section_id="s1")
        assert s.parent_section_id == "s1"


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractionResult:
    """Tests for the ExtractionResult dataclass."""

    def test_construction(self) -> None:
        sec = Section("s1", "", "", "part", 1, 1, "text")
        r = ExtractionResult(source_file="test.pdf", total_pages=5, sections=[sec])
        assert r.source_file == "test.pdf"
        assert r.total_pages == 5
        assert len(r.sections) == 1

    def test_asdict_roundtrip(self) -> None:
        sec = Section("s1", "§1", "Title", "paragraph", 1, 2, "body")
        r = ExtractionResult(source_file="f.pdf", total_pages=3, sections=[sec])
        d = asdict(r)
        assert d["sections"][0]["section_id"] == "s1"


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResult:
    """Tests for the GenerationResult dataclass."""

    def test_construction_with_defaults(self) -> None:
        r = GenerationResult(answer="42", model="llama3.1")
        assert r.answer == "42"
        assert r.model == "llama3.1"
        assert r.context_chunk_ids == []
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0

    def test_context_chunk_ids_default_factory(self) -> None:
        r1 = GenerationResult(answer="a", model="m")
        r2 = GenerationResult(answer="b", model="m")
        assert r1.context_chunk_ids is not r2.context_chunk_ids


# ---------------------------------------------------------------------------
# AbstentionResult
# ---------------------------------------------------------------------------


class TestAbstentionResult:
    """Tests for the AbstentionResult dataclass."""

    def test_answered(self) -> None:
        r = AbstentionResult(
            abstained=False, level=None, reason="",
            answer="The answer.", confidence=0.85,
        )
        assert not r.abstained
        assert r.level is None
        assert r.answer == "The answer."
        assert r.model == ""

    def test_level1_abstention(self) -> None:
        r = AbstentionResult(
            abstained=True, level=1, reason="low confidence",
            answer="", confidence=0.1,
        )
        assert r.abstained
        assert r.level == 1

    def test_retrieval_results_default_factory(self) -> None:
        r1 = AbstentionResult(False, None, "", "a", 0.5)
        r2 = AbstentionResult(False, None, "", "b", 0.5)
        assert r1.retrieval_results is not r2.retrieval_results


# ---------------------------------------------------------------------------
# QualityFlag
# ---------------------------------------------------------------------------


class TestQualityFlag:
    """Tests for the QualityFlag dataclass."""

    def test_construction(self) -> None:
        f = QualityFlag(code="EMPTY_TEXT", message="No text", severity="warning")
        assert f.code == "EMPTY_TEXT"
        assert f.severity == "warning"
        assert f.context == {}

    def test_context_default_factory(self) -> None:
        f1 = QualityFlag("A", "m", "info")
        f2 = QualityFlag("B", "m", "info")
        assert f1.context is not f2.context


# ---------------------------------------------------------------------------
# compute_token_stats
# ---------------------------------------------------------------------------


class TestComputeTokenStats:
    """Tests for compute_token_stats utility."""

    def test_empty_list(self) -> None:
        stats = compute_token_stats([])
        assert stats == {"min_tokens": 0, "median_tokens": 0, "max_tokens": 0, "total_tokens": 0}

    def test_single_value(self) -> None:
        stats = compute_token_stats([100])
        assert stats["min_tokens"] == 100
        assert stats["max_tokens"] == 100
        assert stats["total_tokens"] == 100

    def test_multiple_values(self) -> None:
        stats = compute_token_stats([10, 20, 30])
        assert stats["min_tokens"] == 10
        assert stats["median_tokens"] == 20
        assert stats["max_tokens"] == 30
        assert stats["total_tokens"] == 60


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------


class TestSaveJson:
    """Tests for save_json I/O utility."""

    def test_saves_dataclass(self, tmp_path: Path) -> None:
        result = GenerationResult(answer="test", model="m")
        out = save_json(result, tmp_path / "out.json")
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["answer"] == "test"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "out.json"
        result = QualityFlag("C", "msg", "info")
        out = save_json(result, nested)
        assert out.exists()

    def test_utf8_encoding(self, tmp_path: Path) -> None:
        sec = Section("s1", "§1", "Prüfungsordnung", "paragraph", 1, 1, "Ü ä ö")
        er = ExtractionResult(source_file="f.pdf", total_pages=1, sections=[sec])
        out = save_json(er, tmp_path / "utf8.json")
        text = out.read_text(encoding="utf-8")
        assert "Prüfungsordnung" in text
        assert "\\u" not in text  # ensure_ascii=False
