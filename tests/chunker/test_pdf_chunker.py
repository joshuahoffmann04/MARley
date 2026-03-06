"""Tests for the StPO PDF chunker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tiktoken

from src.marley.extractor import ExtractionResult, Section, Table
from src.marley.chunker import (
    Chunk,
    ChunkingResult,
    ChunkMetadata,
    QualityFlag,
    chunk_stpo,
    save,
)
from src.marley.chunker.pdf_chunker import (
    _apply_heading_prefix,
    _build_heading_prefix,
    _build_table_chunks,
    _merge_undersized,
    _prepare_sentences,
    _serialize_table_row,
    _sliding_window_chunks,
    _split_oversized_sentence,
    _split_sentences,
)

ENCODER = tiktoken.get_encoding("cl100k_base")

EXTRACTED_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "knowledgebase" / "stpo-extracted.json"
)


def _section(
    section_id: str = "par-1",
    kind: str = "paragraph",
    label: str = "§1",
    title: str = "Scope",
    text: str = "This is a test.",
    parent_section_id: str | None = None,
    tables: list[Table] | None = None,
) -> Section:
    return Section(
        section_id=section_id,
        label=label,
        title=title,
        kind=kind,
        start_page=1,
        end_page=1,
        text=text,
        tables=tables or [],
        parent_section_id=parent_section_id,
    )


def _extraction(*sections: Section) -> ExtractionResult:
    return ExtractionResult(
        source_file="test.pdf",
        total_pages=1,
        sections=list(sections),
    )


# ---------------------------------------------------------------------------
# Unit tests — sentence splitting
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_empty_input(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   \n\n  ") == []

    def test_single_sentence(self):
        result = _split_sentences("Hello world.")
        assert len(result) >= 1
        assert "Hello" in result[0]

    def test_multiple_sentences(self):
        result = _split_sentences("First sentence. Second sentence. Third one.")
        assert len(result) >= 2


class TestSplitOversizedSentence:
    def test_normal_sentence_unchanged(self):
        result = _split_oversized_sentence("Short text.", ENCODER, 100)
        assert result == ["Short text."]

    def test_oversized_splits(self):
        long_text = "word " * 200
        result = _split_oversized_sentence(long_text.strip(), ENCODER, 50)
        assert len(result) > 1
        for part in result:
            assert len(ENCODER.encode(part)) <= 50

    def test_empty_input(self):
        result = _split_oversized_sentence("", ENCODER, 100)
        assert result == [""]


# ---------------------------------------------------------------------------
# Unit tests — sentence preparation
# ---------------------------------------------------------------------------

class TestPrepareSentences:
    def test_empty_list(self):
        flat, counts = _prepare_sentences([], ENCODER, 100)
        assert flat == []
        assert counts == []

    def test_normal_sentences(self):
        sentences = ["Hello world.", "Goodbye."]
        flat, counts = _prepare_sentences(sentences, ENCODER, 100)
        assert flat == sentences
        assert len(counts) == 2
        assert all(c > 0 for c in counts)

    def test_oversized_sentence_split(self):
        long_text = "word " * 200
        flat, counts = _prepare_sentences([long_text.strip()], ENCODER, 50)
        assert len(flat) > 1
        assert all(c <= 50 for c in counts)

    def test_counts_match_encoder(self):
        sentences = ["The quick brown fox.", "Lazy dog."]
        flat, counts = _prepare_sentences(sentences, ENCODER, 200)
        for sent, count in zip(flat, counts):
            assert count == len(ENCODER.encode(sent))


# ---------------------------------------------------------------------------
# Unit tests — sliding window
# ---------------------------------------------------------------------------

class TestSlidingWindowChunks:
    def test_empty_input(self):
        assert _sliding_window_chunks([], [], 100, 10) == []

    def test_single_sentence_fits(self):
        result = _sliding_window_chunks(["Hello."], [3], 100, 10)
        assert result == ["Hello."]

    def test_all_sentences_fit_one_chunk(self):
        sentences = ["A.", "B.", "C."]
        counts = [2, 2, 2]
        result = _sliding_window_chunks(sentences, counts, 100, 10)
        assert len(result) == 1
        assert result[0] == "A. B. C."

    def test_overflow_creates_multiple_chunks(self):
        sentences = [f"Sentence {i}." for i in range(10)]
        counts = [len(ENCODER.encode(s)) for s in sentences]
        result = _sliding_window_chunks(sentences, counts, 20, 0)
        assert len(result) > 1

    def test_overlap_repeats_trailing_sentences(self):
        # 5 sentences, ~10 tokens each, window=25, overlap=10
        sentences = ["Alpha bravo charlie.", "Delta echo foxtrot.",
                     "Golf hotel india.", "Juliet kilo lima.",
                     "Mike november oscar."]
        counts = [len(ENCODER.encode(s)) for s in sentences]
        result = _sliding_window_chunks(sentences, counts, 25, 10)
        assert len(result) >= 2
        # The last sentence(s) of chunk 0 should appear at start of chunk 1
        words_0 = result[0].split()
        words_1 = result[1].split()
        # Find shared suffix/prefix
        found_overlap = False
        for n in range(min(len(words_0), len(words_1)), 0, -1):
            if words_0[-n:] == words_1[:n]:
                found_overlap = True
                break
        assert found_overlap, "Expected sentence-aligned overlap between chunks"

    def test_zero_overlap_no_repetition(self):
        sentences = ["First sentence here.", "Second sentence here.",
                     "Third sentence here.", "Fourth sentence here."]
        counts = [len(ENCODER.encode(s)) for s in sentences]
        result = _sliding_window_chunks(sentences, counts, 15, 0)
        assert len(result) >= 2
        # With no overlap, end of chunk N should not equal start of chunk N+1
        all_text = " ".join(result)
        for sent in sentences:
            assert all_text.count(sent) == 1

    def test_oversized_sentence_gets_own_chunk(self):
        sentences = ["Tiny.", "word " * 50]
        counts = [len(ENCODER.encode(s)) for s in sentences]
        result = _sliding_window_chunks(sentences, counts, 30, 10)
        assert len(result) >= 2

    def test_forward_progress_guaranteed(self):
        # Even with large overlap, the window must advance
        sentences = [f"S{i}." for i in range(20)]
        counts = [3] * 20
        result = _sliding_window_chunks(sentences, counts, 10, 8)
        # Must terminate and produce chunks
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Unit tests — merge undersized
# ---------------------------------------------------------------------------

class TestMergeUndersized:
    def test_no_merging_needed(self):
        chunks = ["word " * 30, "text " * 30]
        result = _merge_undersized(
            [c.strip() for c in chunks], ENCODER, min_tokens=10, max_tokens=200,
        )
        assert len(result) == 2

    def test_merge_into_next(self):
        chunks = ["Hi.", "This is a longer piece of text that has enough tokens."]
        result = _merge_undersized(chunks, ENCODER, min_tokens=20, max_tokens=200)
        assert len(result) == 1

    def test_merge_into_prev(self):
        chunks = ["This is a decent chunk with enough tokens for the minimum.", "Hi."]
        result = _merge_undersized(chunks, ENCODER, min_tokens=20, max_tokens=200)
        assert len(result) == 1

    def test_single_chunk(self):
        result = _merge_undersized(["Only one."], ENCODER, min_tokens=20, max_tokens=200)
        assert result == ["Only one."]

    def test_middle_undersized_merged(self):
        # Undersized chunk in the middle should be merged
        chunks = ["word " * 30, "Hi.", "text " * 30]
        result = _merge_undersized(
            [c.strip() for c in chunks], ENCODER, min_tokens=10, max_tokens=400,
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Unit tests — heading prefix
# ---------------------------------------------------------------------------

class TestBuildHeadingPrefix:
    def test_paragraph_with_part_parent(self):
        part = _section("part-III", "part", "III.", "Examination-related provisions")
        par = _section("par-23", "paragraph", "§23", "Master's Thesis",
                       parent_section_id="part-III")
        section_map = {s.section_id: s for s in [part, par]}
        prefix, labels = _build_heading_prefix(par, section_map)
        assert prefix is not None
        assert "III." in prefix
        assert "§23" in prefix
        assert len(labels) == 2

    def test_top_level_section(self):
        part = _section("part-I", "part", "I.", "General")
        section_map = {"part-I": part}
        prefix, labels = _build_heading_prefix(part, section_map)
        assert prefix is not None
        assert "I. General" in prefix
        assert len(labels) == 1

    def test_appendix(self):
        appendix = _section("appendix-2", "appendix", "Appendix 2",
                            "Module List")
        section_map = {"appendix-2": appendix}
        prefix, labels = _build_heading_prefix(appendix, section_map)
        assert "Appendix 2" in prefix

    def test_section_without_label(self):
        sec = _section("preamble", "preamble", "", "")
        section_map = {"preamble": sec}
        prefix, labels = _build_heading_prefix(sec, section_map)
        assert prefix is None
        assert labels == []


# ---------------------------------------------------------------------------
# Unit tests — heading prefix application
# ---------------------------------------------------------------------------

class TestApplyHeadingPrefix:
    def test_no_heading(self):
        result = _apply_heading_prefix(
            ["Hello world."], ENCODER, max_tokens=100, heading_prefix=None,
        )
        assert result == ["Hello world."]

    def test_heading_prepended(self):
        result = _apply_heading_prefix(
            ["Some content here."], ENCODER, max_tokens=200,
            heading_prefix="§23 Master's Thesis\n\n",
        )
        assert len(result) == 1
        assert "§23" in result[0]
        assert "Some content" in result[0]

    def test_empty_chunks(self):
        result = _apply_heading_prefix(
            [], ENCODER, max_tokens=100, heading_prefix=None,
        )
        assert result == []

    def test_body_truncated_to_budget(self):
        long_body = "word " * 200
        result = _apply_heading_prefix(
            [long_body.strip()], ENCODER, max_tokens=50,
            heading_prefix="Heading\n\n",
        )
        assert len(result) == 1
        assert len(ENCODER.encode(result[0])) <= 50


# ---------------------------------------------------------------------------
# Unit tests — table serialization and chunking
# ---------------------------------------------------------------------------

class TestSerializeTableRow:
    def test_normal_row(self):
        result = _serialize_table_row(["CS 627", "9", "elective"])
        assert result == "CS 627 | 9 | elective"

    def test_row_with_empty_cells(self):
        result = _serialize_table_row(["CS 627", "", "elective", ""])
        assert result == "CS 627 | elective"


class TestBuildTableChunks:
    def test_small_table_single_chunk(self):
        table = Table(
            table_id="tbl-1", page=1,
            headers=["Name", "LP"],
            rows=[["CS 627", "9"], ["CS 561", "6"]],
        )
        result = _build_table_chunks(table, ENCODER, max_tokens=200, heading_prefix=None)
        assert len(result) == 1
        assert "Name | LP" in result[0]
        assert "CS 627" in result[0]

    def test_large_table_multiple_chunks(self):
        rows = [[f"Module {i}", str(i)] for i in range(100)]
        table = Table(table_id="tbl-1", page=1, headers=["Name", "LP"], rows=rows)
        result = _build_table_chunks(table, ENCODER, max_tokens=50, heading_prefix=None)
        assert len(result) > 1
        for chunk in result[:-1]:
            assert "Name | LP" in chunk

    def test_empty_table(self):
        table = Table(table_id="tbl-1", page=1, headers=["Name"], rows=[])
        result = _build_table_chunks(table, ENCODER, max_tokens=200, heading_prefix=None)
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests — chunk IDs
# ---------------------------------------------------------------------------

class TestChunkId:
    def test_text_chunk_id_format(self):
        sec = _section(text="Hello world. " * 10)
        result = chunk_stpo(_extraction(sec), max_chunk_tokens=512)
        text_chunks = [c for c in result.chunks if c.chunk_type == "text"]
        assert text_chunks
        assert text_chunks[0].chunk_id == "par-1-txt-1"

    def test_table_chunk_id_format(self):
        table = Table(table_id="par-1-tbl-1", page=1,
                      headers=["A", "B"], rows=[["x", "y"]])
        sec = _section(text="Some text.", tables=[table])
        result = chunk_stpo(_extraction(sec), max_chunk_tokens=512)
        table_chunks = [c for c in result.chunks if c.chunk_type == "table"]
        assert table_chunks
        assert "tbl-" in table_chunks[0].chunk_id
        assert "par-1" in table_chunks[0].chunk_id


# ---------------------------------------------------------------------------
# Integration tests — require stpo-extracted.json
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.skipif(
    not EXTRACTED_PATH.exists(),
    reason="stpo-extracted.json not found",
)


def _load_extraction() -> ExtractionResult:
    data = json.loads(EXTRACTED_PATH.read_text(encoding="utf-8"))
    sections = []
    for s in data["sections"]:
        tables = [
            Table(
                table_id=t["table_id"],
                page=t["page"],
                headers=t["headers"],
                rows=t["rows"],
            )
            for t in s.get("tables", [])
        ]
        sections.append(Section(
            section_id=s["section_id"],
            label=s["label"],
            title=s["title"],
            kind=s["kind"],
            start_page=s["start_page"],
            end_page=s["end_page"],
            text=s["text"],
            tables=tables,
            parent_section_id=s.get("parent_section_id"),
        ))
    return ExtractionResult(
        source_file=data["source_file"],
        total_pages=data["total_pages"],
        sections=sections,
    )


@pytest.fixture(scope="module")
def extraction() -> ExtractionResult:
    return _load_extraction()


@pytest.fixture(scope="module")
def chunking_result(extraction: ExtractionResult) -> ChunkingResult:
    return chunk_stpo(extraction)


@pytestmark_integration
class TestChunkingBasics:
    def test_total_chunks_positive(self, chunking_result):
        assert chunking_result.stats.total_chunks > 0

    def test_all_chunks_have_text(self, chunking_result):
        for chunk in chunking_result.chunks:
            assert chunk.text.strip()

    def test_all_chunks_have_metadata(self, chunking_result):
        for chunk in chunking_result.chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.section_id is not None


@pytestmark_integration
class TestTokenBounds:
    def test_no_chunk_exceeds_max(self, chunking_result):
        for chunk in chunking_result.chunks:
            assert chunk.token_count <= 512

    def test_stats_match_chunks(self, chunking_result):
        assert chunking_result.stats.total_chunks == len(chunking_result.chunks)
        assert chunking_result.stats.text_chunks == sum(
            1 for c in chunking_result.chunks if c.chunk_type == "text"
        )
        assert chunking_result.stats.table_chunks == sum(
            1 for c in chunking_result.chunks if c.chunk_type == "table"
        )

    def test_token_stats_consistent(self, chunking_result):
        tokens = [c.token_count for c in chunking_result.chunks]
        assert chunking_result.stats.min_tokens == min(tokens)
        assert chunking_result.stats.max_tokens == max(tokens)
        assert chunking_result.stats.total_tokens == sum(tokens)


@pytestmark_integration
class TestSlidingWindowOverlap:
    """Verify that consecutive text chunks share sentence-aligned overlap."""

    def test_multi_chunk_sections_have_overlap(self, chunking_result):
        from collections import Counter
        section_ids = [
            c.metadata.section_id for c in chunking_result.chunks
            if c.chunk_type == "text"
        ]
        multi = [sid for sid, n in Counter(section_ids).items() if n >= 3]
        assert multi, "Expected at least one section with 3+ text chunks"

        for sid in multi:
            chunks = [
                c for c in chunking_result.chunks
                if c.metadata.section_id == sid and c.chunk_type == "text"
            ]
            for i in range(len(chunks) - 1):
                body_a = chunks[i].text.split("\n\n", 1)[-1]
                body_b = chunks[i + 1].text.split("\n\n", 1)[-1]
                words_a = body_a.split()
                words_b = body_b.split()
                overlap_found = any(
                    words_a[-n:] == words_b[:n]
                    for n in range(min(80, len(words_a), len(words_b)), 0, -1)
                )
                # Overlap may be absent when a single sentence fills the budget
                if not overlap_found:
                    # Verify this is the oversized-sentence edge case
                    assert (
                        chunks[i].token_count > 400
                        or chunks[i + 1].token_count > 400
                    ), (
                        f"Missing overlap in {sid} chunks {i+1}->{i+2} "
                        f"without oversized sentence justification"
                    )


@pytestmark_integration
class TestSectionCoverage:
    def test_all_sections_produce_chunks(self, extraction, chunking_result):
        chunked_sections = {c.metadata.section_id for c in chunking_result.chunks}
        for section in extraction.sections:
            if section.text.strip() or section.tables:
                assert section.section_id in chunked_sections, (
                    f"Section {section.section_id} produced no chunks"
                )

    def test_paragraph_chunks_have_heading(self, chunking_result):
        par_text_chunks = [
            c for c in chunking_result.chunks
            if c.metadata.section_kind == "paragraph" and c.chunk_type == "text"
        ]
        assert par_text_chunks
        for chunk in par_text_chunks:
            assert chunk.metadata.heading_path
            assert len(chunk.metadata.heading_path) >= 2

    def test_all_section_kinds_chunked(self, chunking_result):
        kinds = {c.metadata.section_kind for c in chunking_result.chunks}
        assert "paragraph" in kinds
        assert "appendix" in kinds
        assert "part" in kinds


@pytestmark_integration
class TestTableChunking:
    def test_appendix_2_produces_table_chunks(self, chunking_result):
        a2_tables = [
            c for c in chunking_result.chunks
            if c.chunk_type == "table" and c.metadata.section_id == "appendix-2"
        ]
        assert len(a2_tables) >= 1

    def test_table_chunks_repeat_headers(self, chunking_result):
        a2_tables = [
            c for c in chunking_result.chunks
            if c.chunk_type == "table" and c.metadata.section_id == "appendix-2"
        ]
        if len(a2_tables) > 1:
            for chunk in a2_tables:
                assert "Name of module" in chunk.text or "LP" in chunk.text

    def test_table_chunk_ids_contain_section(self, chunking_result):
        for chunk in chunking_result.chunks:
            if chunk.chunk_type == "table":
                assert chunk.metadata.section_id in chunk.chunk_id

    def test_table_metadata_has_table_id(self, chunking_result):
        for chunk in chunking_result.chunks:
            if chunk.chunk_type == "table":
                assert chunk.metadata.table_id is not None


@pytestmark_integration
class TestHeadingPaths:
    def test_paragraph_includes_part_in_path(self, chunking_result):
        par_chunks = [
            c for c in chunking_result.chunks
            if c.metadata.section_kind == "paragraph"
        ]
        assert par_chunks
        for chunk in par_chunks:
            path_text = " ".join(chunk.metadata.heading_path)
            has_part = any(
                numeral in path_text for numeral in ["I.", "II.", "III.", "IV."]
            )
            assert has_part, f"Paragraph chunk missing part in path: {chunk.metadata.heading_path}"

    def test_appendix_has_correct_path(self, chunking_result):
        app_chunks = [
            c for c in chunking_result.chunks
            if c.metadata.section_kind == "appendix"
        ]
        assert app_chunks
        for chunk in app_chunks:
            assert any("Appendix" in label for label in chunk.metadata.heading_path)

    def test_preamble_path(self, chunking_result):
        preamble_chunks = [
            c for c in chunking_result.chunks
            if c.metadata.section_id == "preamble"
        ]
        assert preamble_chunks
        for chunk in preamble_chunks:
            assert chunk.metadata.section_kind == "preamble"


@pytestmark_integration
class TestQualityFlags:
    def test_no_error_flags(self, chunking_result):
        errors = [f for f in chunking_result.quality_flags if f.severity == "error"]
        assert not errors, f"Unexpected error flags: {errors}"

    def test_stats_match_actual(self, chunking_result):
        assert chunking_result.stats.sections_processed > 0
        assert chunking_result.stats.tables_processed > 0


@pytestmark_integration
class TestSaveAndLoad:
    def test_json_roundtrip(self, chunking_result, tmp_path):
        out = save(chunking_result, tmp_path / "chunks.json")
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["stats"]["total_chunks"] == chunking_result.stats.total_chunks
        assert len(data["chunks"]) == len(chunking_result.chunks)

    def test_save_creates_parent_dirs(self, chunking_result, tmp_path):
        out = save(chunking_result, tmp_path / "nested" / "dir" / "out.json")
        assert out.exists()
