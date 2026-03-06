"""Tests for the StPO PDF extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.marley.extractor import ExtractionResult, Section, Table, extract, save
from src.marley.extractor.extractor import (
    _assign_parents, _cell_text, _is_continuation_row, _is_header_row,
    _is_section_label_row, _make_section_id, _Marker,
    _merge_appendix2_continuations, _merge_continuation,
    _normalize_whitespace, _strip_page_number, extract_page_texts,
)

PDF_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "msc-computer-science.pdf"
pytestmark = pytest.mark.skipif(not PDF_PATH.exists(), reason="StPO PDF not found")


@pytest.fixture(scope="module")
def result() -> ExtractionResult:
    return extract(PDF_PATH)


@pytest.fixture(scope="module")
def sections(result: ExtractionResult) -> list[Section]:
    return result.sections


@pytest.fixture(scope="module")
def section_map(sections: list[Section]) -> dict[str, Section]:
    return {s.section_id: s for s in sections}



class TestStripPageNumber:
    def test_page_1_unchanged(self):
        assert _strip_page_number("Hello\nWorld", 1) == "Hello\nWorld"

    def test_strips_leading_number(self):
        assert _strip_page_number("3\nContent here", 3) == "Content here"

    def test_no_number_unchanged(self):
        assert _strip_page_number("Content here", 5) == "Content here"


class TestNormalizeWhitespace:
    def test_collapses_blank_lines(self):
        assert _normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_strips_trailing_ws(self):
        assert _normalize_whitespace("hello   \nworld  ") == "hello\nworld"


class TestCellText:
    def test_none_returns_empty(self):
        assert _cell_text(None) == ""

    def test_strips_and_replaces_newlines(self):
        assert _cell_text("  hello\nworld  ") == "hello world"


class TestIsHeaderRow:
    def test_detects_header(self):
        row = ["Name of module", None, None, "LP", None, None, None, None, None, None, None, None, None]
        assert _is_header_row(row) is True

    def test_normal_row(self):
        row = ["CS 627", None, None, "9", None, None, None, None, None, None, None, None, None]
        assert _is_header_row(row) is False


class TestIsSectionLabelRow:
    def test_section_label_detected(self):
        row = ["Compulsory Elective Modules", None, None, None, None, None, None]
        assert _is_section_label_row(row) is True

    def test_module_row_not_label(self):
        row = ["CS 627 Advanced Algorithms", None, None, "9", None, None, None]
        assert _is_section_label_row(row) is False

    def test_short_text_not_label(self):
        row = ["Short", None, None, None, None, None, None]
        assert _is_section_label_row(row) is False

    def test_multiple_non_empty_not_label(self):
        row = ["Some text here", "9", None, None, None, None, None]
        assert _is_section_label_row(row) is False


class TestIsContinuationRow:
    def test_with_lp_not_continuation(self):
        assert _is_continuation_row(["CS 627", "9", "x", "", "", "", ""]) is False

    def test_no_lp_with_text_is_continuation(self):
        assert _is_continuation_row(["overflow", "", "", "", "more", "", ""]) is True

    def test_all_empty_not_continuation(self):
        assert _is_continuation_row(["", "", "", "", "", "", ""]) is False


class TestMergeContinuation:
    def test_appends_text(self):
        parent = ["CS 624", "9", "elective", "Advanced", "goals", "None.", "credit"]
        cont = ["overflow", "", "", "in theory", "more goals", "", "more credit"]
        _merge_continuation(parent, cont)
        assert parent[0] == "CS 624 overflow"
        assert parent[3] == "Advanced in theory"


class TestMergeAppendix2Continuations:
    def test_merges_correctly(self):
        rows = [
            ["CS 624", "9", "e", "Adv", "g", "N", "c"],
            ["ov", "", "", "th", "m", "", ""],
            ["CS 561", "9", "e", "Adv", "o", "N", "o"],
        ]
        merged = _merge_appendix2_continuations(rows)
        assert len(merged) == 2
        assert merged[0][0] == "CS 624 ov"
        assert merged[1][0] == "CS 561"


class TestMakeSectionId:
    def test_preamble(self):
        assert _make_section_id(_Marker("preamble", "Preamble", "", 1, 0)) == "preamble"

    def test_paragraph(self):
        assert _make_section_id(_Marker("paragraph", "§23", "", 11, 0)) == "par-23"

    def test_appendix(self):
        assert _make_section_id(_Marker("appendix", "Appendix 2", "", 20, 0)) == "appendix-2"

def _section(kind: str, section_id: str) -> Section:
    """Create a minimal Section for unit testing."""
    return Section(
        section_id=section_id,
        label=section_id,
        title="",
        kind=kind,
        start_page=1,
        end_page=1,
        text="",
    )


class TestAssignParents:
    def test_paragraph_gets_part_parent(self):
        sections = [_section("part", "part-I"), _section("paragraph", "par-1")]
        _assign_parents(sections)
        assert sections[1].parent_section_id == "part-I"

    def test_part_has_no_parent(self):
        sections = [_section("part", "part-I")]
        _assign_parents(sections)
        assert sections[0].parent_section_id is None

    def test_preamble_has_no_parent(self):
        sections = [_section("preamble", "preamble")]
        _assign_parents(sections)
        assert sections[0].parent_section_id is None

    def test_toc_has_no_parent(self):
        sections = [_section("toc", "toc")]
        _assign_parents(sections)
        assert sections[0].parent_section_id is None

    def test_appendix_has_no_parent(self):
        sections = [_section("part", "part-IV"), _section("appendix", "appendix-1")]
        _assign_parents(sections)
        assert sections[1].parent_section_id is None

    def test_part_switch_updates_parent(self):
        sections = [
            _section("part", "part-I"),
            _section("paragraph", "par-1"),
            _section("part", "part-II"),
            _section("paragraph", "par-4"),
        ]
        _assign_parents(sections)
        assert sections[1].parent_section_id == "part-I"
        assert sections[3].parent_section_id == "part-II"

    def test_empty_list(self):
        sections: list[Section] = []
        _assign_parents(sections)
        assert sections == []


class TestExtractionBasics:
    def test_total_pages(self, result):
        assert result.total_pages == 47

    def test_source_file(self, result):
        assert "msc-computer-science.pdf" in result.source_file

    def test_section_count(self, sections):
        assert len(sections) == 48


class TestSectionDetection:
    def test_preamble_exists(self, section_map):
        s = section_map["preamble"]
        assert s.kind == "preamble"
        assert s.start_page == 1

    def test_toc_exists(self, section_map):
        s = section_map["toc"]
        assert s.kind == "toc"

    def test_all_four_parts(self, section_map):
        for numeral in ["I", "II", "III", "IV"]:
            assert f"part-{numeral}" in section_map

    def test_all_paragraphs_1_to_38(self, section_map):
        for num in range(1, 39):
            assert f"par-{num}" in section_map

    def test_section_kinds(self, sections):
        kinds = {s.kind for s in sections}
        assert kinds == {"preamble", "toc", "part", "paragraph", "appendix"}

    def test_paragraph_36_detected(self, section_map):
        assert "par-36" in section_map

    def test_no_duplicate_sections(self, sections):
        ids = [s.section_id for s in sections]
        assert len(ids) == len(set(ids))



class TestSectionContent:
    def test_preamble_has_text(self, section_map):
        assert len(section_map["preamble"].text) > 50

    def test_paragraph_text_not_empty(self, section_map):
        for num in range(1, 39):
            s = section_map[f"par-{num}"]
            assert s.text

    def test_paragraph_23_mentions_thesis(self, section_map):
        text = section_map["par-23"].text.lower()
        assert "thesis" in text or "master" in text


class TestParentAssignment:
    def test_par_1_to_3_parent_is_part_I(self, section_map):
        for num in range(1, 4):
            assert section_map[f"par-{num}"].parent_section_id == "part-I"

    def test_par_4_to_15_parent_is_part_II(self, section_map):
        for num in range(4, 16):
            assert section_map[f"par-{num}"].parent_section_id == "part-II"

    def test_par_16_to_36_parent_is_part_III(self, section_map):
        for num in range(16, 37):
            assert section_map[f"par-{num}"].parent_section_id == "part-III"

    def test_par_37_38_parent_is_part_IV(self, section_map):
        for num in range(37, 39):
            assert section_map[f"par-{num}"].parent_section_id == "part-IV"

    def test_parts_have_no_parent(self, section_map):
        for numeral in ["I", "II", "III", "IV"]:
            assert section_map[f"part-{numeral}"].parent_section_id is None

    def test_appendices_have_no_parent(self, section_map):
        for num in range(1, 5):
            assert section_map[f"appendix-{num}"].parent_section_id is None

    def test_preamble_and_toc_have_no_parent(self, section_map):
        assert section_map["preamble"].parent_section_id is None
        assert section_map["toc"].parent_section_id is None

    def test_save_includes_parent_section_id(self, result, tmp_path):
        out = save(result, tmp_path / "parent-test.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        par_1 = next(s for s in data["sections"] if s["section_id"] == "par-1")
        assert par_1["parent_section_id"] == "part-I"
        appendix = next(s for s in data["sections"] if s["section_id"] == "appendix-1")
        assert appendix["parent_section_id"] is None


class TestPageRanges:
    def test_sections_cover_all_pages(self, sections):
        covered = set()
        for s in sections:
            for pg in range(s.start_page, s.end_page + 1):
                covered.add(pg)
        assert covered == set(range(1, 48))

    def test_appendix_2_spans_many_pages(self, section_map):
        s = section_map["appendix-2"]
        assert s.end_page - s.start_page >= 10


class TestTableExtraction:
    def test_total_tables(self, sections):
        total = sum(len(s.tables) for s in sections)
        assert total >= 20

    def test_appendix_2_has_one_table(self, section_map):
        assert len(section_map["appendix-2"].tables) == 1

    def test_appendix_2_table_headers(self, section_map):
        tbl = section_map["appendix-2"].tables[0]
        assert len(tbl.headers) == 7
        assert tbl.headers[0] == "Name of module / German translation"
        assert tbl.headers[1] == "LP"

    def test_appendix_2_module_count(self, section_map):
        assert len(section_map["appendix-2"].tables[0].rows) == 54

    def test_appendix_2_cs_module_count(self, section_map):
        tbl = section_map["appendix-2"].tables[0]
        cs = [r for r in tbl.rows if r[0].startswith("CS ")]
        assert len(cs) == 46

    def test_appendix_2_conditional_count(self, section_map):
        tbl = section_map["appendix-2"].tables[0]
        cond = [r for r in tbl.rows if r[0].startswith("Conditional")]
        assert len(cond) == 8

    def test_appendix_2_no_empty_rows(self, section_map):
        for row in section_map["appendix-2"].tables[0].rows:
            assert row[0].strip()
            assert row[1].strip()

    def test_appendix_2_lp_numeric(self, section_map):
        for row in section_map["appendix-2"].tables[0].rows:
            assert row[1].strip().isdigit()

    def test_appendix_3_has_tables(self, section_map):
        assert len(section_map["appendix-3"].tables) >= 10

    def test_appendix_4_has_tables(self, section_map):
        assert len(section_map["appendix-4"].tables) >= 1

    def test_table_ids_unique(self, sections):
        ids = [t.table_id for s in sections for t in s.tables]
        assert len(ids) == len(set(ids))

    def test_table_ids_contain_section_id(self, sections):
        for s in sections:
            for t in s.tables:
                assert s.section_id in t.table_id


class TestSaveAndLoad:
    def test_save_roundtrip(self, result, tmp_path):
        out = save(result, tmp_path / "test-output.json")
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_pages"] == 47
        assert len(data["sections"]) == 48

    def test_save_creates_parent_dirs(self, result, tmp_path):
        out = save(result, tmp_path / "nested" / "dir" / "out.json")
        assert out.exists()


class TestExtractErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract("nonexistent.pdf")
