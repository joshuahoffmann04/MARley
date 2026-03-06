"""PDF extractor for the StPO document.

Extracts the StPO PDF into structured sections with text and tables,
producing a JSON file for downstream chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz
import pdfplumber

from src.marley.models import ExtractionResult, Section, Table, save_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION_PATTERN = re.compile(r"^\s*\u00a7\s*(\d+[a-z]?)\s*$")
APPENDIX_PATTERN = re.compile(r"^\s*Appendix\s+(\d+)\s*:\s*(.*)", re.IGNORECASE)
ROMAN_PATTERN = re.compile(r"^\s*([IVXLC]+)\.\s*$")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,2}\s*$")
TOC_MARKER = "table of contents"

# Appendix 2 has 13 pdfplumber columns; these indices map to the 7 real columns.
APPENDIX2_COL_INDICES = [0, 3, 4, 7, 8, 9, 10]
APPENDIX2_HEADERS = [
    "Name of module / German translation",
    "LP",
    "Degree of obligation",
    "Level",
    "Qualification goals",
    "Prerequisites",
    "Prerequisites to earn credits (LP)",
]


# ---------------------------------------------------------------------------
# Text extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def _strip_page_number(text: str, page_number: int) -> str:
    """Remove the leading page-number line from extracted text."""
    if page_number == 1:
        return text
    lines = text.split("\n")
    if lines and PAGE_NUMBER_PATTERN.match(lines[0]):
        return "\n".join(lines[1:])
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines into single blank lines, strip trailing ws."""
    lines = [line.rstrip() for line in text.split("\n")]
    result: list[str] = []
    prev_blank = False
    for line in lines:
        if not line.strip():
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return "\n".join(result).strip()


def extract_page_texts(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract cleaned text for each page. Returns list of (page_number, text)."""
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            page_number = i + 1
            raw = page.get_text("text") or ""
            cleaned = _strip_page_number(raw, page_number)
            cleaned = _normalize_whitespace(cleaned)
            pages.append((page_number, cleaned))
    return pages


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

@dataclass
class _Marker:
    """Internal marker for a detected section boundary."""
    kind: str
    label: str
    title: str
    page: int
    line_index: int


def _detect_markers(pages: list[tuple[int, str]]) -> list[_Marker]:
    """Scan all pages for section-starting markers."""
    markers: list[_Marker] = []

    for page_number, text in pages:
        lines = text.split("\n")

        # Page 1: preamble
        if page_number == 1:
            markers.append(_Marker(
                kind="preamble",
                label="Preamble",
                title="Degree Program and Examination Regulations",
                page=page_number,
                line_index=0,
            ))
            continue

        # ToC detection
        if any(TOC_MARKER in line.lower() for line in lines):
            markers.append(_Marker(
                kind="toc",
                label="Table of Contents",
                title="Table of Contents",
                page=page_number,
                line_index=0,
            ))
            continue

        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # Roman numeral parts: I., II., III., IV.
            roman_match = ROMAN_PATTERN.match(stripped)
            if roman_match:
                numeral = roman_match.group(1)
                title = _next_non_empty_line(lines, line_idx)
                markers.append(_Marker(
                    kind="part",
                    label=f"{numeral}.",
                    title=title,
                    page=page_number,
                    line_index=line_idx,
                ))
                continue

            # Paragraphs
            section_match = SECTION_PATTERN.match(stripped)
            if section_match:
                num = section_match.group(1)
                title = _next_non_empty_line(lines, line_idx)
                markers.append(_Marker(
                    kind="paragraph",
                    label=f"\u00a7{num}",
                    title=title,
                    page=page_number,
                    line_index=line_idx,
                ))
                continue

            # Appendix headers
            appendix_match = APPENDIX_PATTERN.match(stripped)
            if appendix_match:
                num = appendix_match.group(1)
                title = appendix_match.group(2).strip()
                if not title:
                    title = _next_non_empty_line(lines, line_idx)
                markers.append(_Marker(
                    kind="appendix",
                    label=f"Appendix {num}",
                    title=title,
                    page=page_number,
                    line_index=line_idx,
                ))
                continue

    # Deduplicate: if two consecutive markers have the same label+kind,
    # keep only the first (e.g., "Appendix 1" appears on pages 17 and 18).
    deduped: list[_Marker] = []
    seen_labels: set[tuple[str, str]] = set()
    for m in markers:
        key = (m.kind, m.label)
        if key in seen_labels:
            continue
        seen_labels.add(key)
        deduped.append(m)
    return deduped


def _next_non_empty_line(lines: list[str], current_index: int) -> str:
    """Return the next non-empty line after current_index."""
    for i in range(current_index + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped and not PAGE_NUMBER_PATTERN.match(stripped):
            return stripped
    return ""


# ---------------------------------------------------------------------------
# Section assembly
# ---------------------------------------------------------------------------

def _build_sections(
    pages: list[tuple[int, str]],
    markers: list[_Marker],
) -> list[Section]:
    """Assemble full text for each section from marker boundaries."""
    if not markers:
        return []

    page_map: dict[int, str] = {p: t for p, t in pages}
    all_page_numbers = [p for p, _ in pages]

    sections: list[Section] = []
    for i, marker in enumerate(markers):
        if i + 1 < len(markers):
            next_marker = markers[i + 1]
            if next_marker.page == marker.page:
                end_page = marker.page
            else:
                end_page = next_marker.page - 1
        else:
            end_page = all_page_numbers[-1]

        text_parts: list[str] = []
        for pg in range(marker.page, end_page + 1):
            page_text = page_map.get(pg, "")
            if pg == marker.page and marker.line_index > 0:
                lines = page_text.split("\n")
                page_text = "\n".join(lines[marker.line_index:])
            if i + 1 < len(markers):
                next_m = markers[i + 1]
                if pg == next_m.page and next_m.line_index > 0:
                    lines = page_text.split("\n")
                    page_text = "\n".join(lines[:next_m.line_index])
            text_parts.append(page_text)

        full_text = _normalize_whitespace("\n".join(text_parts))

        section_id = _make_section_id(marker)
        sections.append(Section(
            section_id=section_id,
            label=marker.label,
            title=marker.title,
            kind=marker.kind,
            start_page=marker.page,
            end_page=end_page,
            text=full_text,
            tables=[],
        ))

    return sections


def _make_section_id(marker: _Marker) -> str:
    """Generate a section ID from a marker."""
    if marker.kind == "preamble":
        return "preamble"
    if marker.kind == "toc":
        return "toc"
    if marker.kind == "part":
        return f"part-{marker.label.rstrip('.')}"
    if marker.kind == "paragraph":
        num = marker.label.replace("\u00a7", "").strip()
        return f"par-{num}"
    if marker.kind == "appendix":
        num = marker.label.replace("Appendix ", "")
        return f"appendix-{num}"
    return marker.label.lower().replace(" ", "-")


# ---------------------------------------------------------------------------
# Parent assignment
# ---------------------------------------------------------------------------

def _assign_parents(sections: list[Section]) -> None:
    """Assign each section its parent based on the document hierarchy.

    Parts act as containers for paragraphs. The most recently encountered
    part becomes the parent for all subsequent paragraphs until the next
    part appears. Preamble, ToC, parts themselves, and appendices are
    top-level sections with no parent.
    """
    current_part_id: str | None = None

    for section in sections:
        if section.kind == "part":
            current_part_id = section.section_id
            section.parent_section_id = None
        elif section.kind == "paragraph":
            section.parent_section_id = current_part_id
        else:
            section.parent_section_id = None


# ---------------------------------------------------------------------------
# Table extraction (pdfplumber)
# ---------------------------------------------------------------------------

def _extract_all_tables(pdf_path: Path) -> list[Table]:
    """Extract all tables from the PDF using pdfplumber.

    Appendix 2 tables (13-column format) are merged into a single table
    across all pages. Other tables are processed individually.
    """
    tables: list[Table] = []
    table_counter = 0

    # Collect all Appendix 2 rows across pages for merging
    appendix2_rows: list[list[str]] = []
    appendix2_start_page: int | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_number = page_idx + 1
            found = page.find_tables()
            if not found:
                continue

            for tbl in found:
                raw_rows = tbl.extract()
                if not raw_rows or not raw_rows[0]:
                    continue

                num_cols = len(raw_rows[0])

                if num_cols == 13:
                    # Accumulate Appendix 2 rows for later merging
                    if appendix2_start_page is None:
                        appendix2_start_page = page_number
                    rows = _extract_appendix2_rows(raw_rows)
                    appendix2_rows.extend(rows)
                else:
                    table_counter += 1
                    table_id = f"tbl-{table_counter:03d}"
                    processed = _process_generic_table(raw_rows, page_number, table_id)
                    if processed:
                        tables.append(processed)

    # Build the single merged Appendix 2 table
    if appendix2_rows and appendix2_start_page is not None:
        merged_rows = _merge_appendix2_continuations(appendix2_rows)
        table_counter += 1
        table_id = f"tbl-{table_counter:03d}"
        tables.append(Table(
            table_id=table_id,
            page=appendix2_start_page,
            headers=list(APPENDIX2_HEADERS),
            rows=merged_rows,
        ))

    return tables


def _cell_text(cell: str | None) -> str:
    """Clean a single cell value."""
    if cell is None:
        return ""
    return cell.replace("\n", " ").strip()


def _is_header_row(row: list[str | None]) -> bool:
    """Detect if a row is a repeated header in Appendix 2."""
    joined = " ".join(_cell_text(c) for c in row)
    return "Name of module" in joined or "German translation" in joined or "Deutscher Modultitel" in joined


def _is_section_label_row(row: list[str | None]) -> bool:
    """Detect section separator rows like 'Compulsory Elective Modules'."""
    non_empty = [c for c in row if c and c.strip()]
    if len(non_empty) == 1:
        text = non_empty[0].strip()
        if len(text) > 10 and not re.match(r"^CS\s+\d+", text):
            return True
    return False


def _is_continuation_row(mapped: list[str]) -> bool:
    """Detect if a mapped 7-column row is a continuation of the previous row.

    Every real module row has an LP value. A row with text but no LP is
    overflow from the previous row's multi-line cells.
    """
    lp = mapped[1].strip()
    if lp:
        return False
    # Has at least some text content (not fully empty)
    return any(cell.strip() for cell in mapped)


def _merge_continuation(parent: list[str], cont: list[str]) -> None:
    """Merge a continuation row into its parent row by appending text."""
    for i in range(len(parent)):
        extra = cont[i].strip() if i < len(cont) else ""
        if extra:
            if parent[i] and not parent[i].endswith(" "):
                parent[i] += " "
            parent[i] += extra


def _extract_appendix2_rows(
    raw_rows: list[list[str | None]],
) -> list[list[str]]:
    """Extract data rows from a 13-column Appendix 2 page table.

    Filters out header rows, section labels, and empty rows.
    Maps 13 columns to the canonical 7-column format.
    """
    data_rows: list[list[str]] = []

    for row in raw_rows:
        if _is_header_row(row):
            continue
        if _is_section_label_row(row):
            continue

        mapped = []
        for col_idx in APPENDIX2_COL_INDICES:
            mapped.append(_cell_text(row[col_idx] if col_idx < len(row) else None))

        if not any(cell for cell in mapped):
            continue

        data_rows.append(mapped)

    return data_rows


def _merge_appendix2_continuations(rows: list[list[str]]) -> list[list[str]]:
    """Merge continuation rows into their parent across all collected pages.

    A continuation row has text but no LP value (col 1). Its cell contents
    are appended to the previous real module row.
    """
    merged: list[list[str]] = []

    for row in rows:
        if _is_continuation_row(row) and merged:
            _merge_continuation(merged[-1], row)
        else:
            merged.append(row)

    return merged


def _process_generic_table(
    raw_rows: list[list[str | None]],
    page_number: int,
    table_id: str,
) -> Table | None:
    """Process a generic table. First non-empty row becomes headers."""
    cleaned_rows: list[list[str]] = []
    for row in raw_rows:
        cleaned = [_cell_text(c) for c in row]
        if any(cell for cell in cleaned):
            cleaned_rows.append(cleaned)

    if not cleaned_rows:
        return None

    # Remove columns that are empty across ALL rows
    num_cols = len(cleaned_rows[0])
    non_empty_cols = [
        col_idx for col_idx in range(num_cols)
        if any(row[col_idx] for row in cleaned_rows if col_idx < len(row))
    ]
    cleaned_rows = [
        [row[col_idx] for col_idx in non_empty_cols if col_idx < len(row)]
        for row in cleaned_rows
    ]

    if not cleaned_rows:
        return None

    headers = cleaned_rows[0]
    data_rows = cleaned_rows[1:]

    return Table(
        table_id=table_id,
        page=page_number,
        headers=headers,
        rows=data_rows,
    )


# ---------------------------------------------------------------------------
# Table assignment to sections
# ---------------------------------------------------------------------------

def _assign_tables(sections: list[Section], tables: list[Table]) -> None:
    """Assign each table to the section whose page range contains it."""
    for table in tables:
        for section in reversed(sections):
            if section.start_page <= table.page <= section.end_page:
                section_tbl_count = len(section.tables) + 1
                table.table_id = f"{section.section_id}-tbl-{section_tbl_count}"
                section.tables.append(table)
                break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(pdf_path: str | Path) -> ExtractionResult:
    """Extract the StPO PDF into structured sections with text and tables."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_page_texts(pdf_path)
    markers = _detect_markers(pages)
    sections = _build_sections(pages, markers)
    _assign_parents(sections)
    tables = _extract_all_tables(pdf_path)
    _assign_tables(sections, tables)

    return ExtractionResult(
        source_file=str(pdf_path),
        total_pages=len(pages),
        sections=sections,
    )


def save(result: ExtractionResult, output_path: str | Path) -> Path:
    """Save an ExtractionResult as JSON."""
    return save_json(result, output_path)
