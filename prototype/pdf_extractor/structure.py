from __future__ import annotations

import re
from dataclasses import dataclass

from pdf_extractor.models import QualityFlag, SectionRecord, TOCEntry

ROMAN_ONLY_PATTERN = re.compile(r"^\s*([IVXLCDM]{1,8})\.\s*$")
ROMAN_INLINE_PATTERN = re.compile(r"^\s*([IVXLCDM]{1,8})\.\s+(.+?)\s*$")
ANNEX_PATTERN = re.compile(r"^\s*Anlage\s+(\d+)\s*:?\s*(.*)\s*$", re.IGNORECASE)
PARAGRAPH_PATTERN = re.compile(r"^\s*§+\s*(\d+[a-zA-Z]?)\s*(.*)$")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*$")
PARAGRAPH_REFERENCE_HINTS = (
    "abs.",
    "satz",
    "nr.",
    "gemäß",
    "anlage",
)


@dataclass(frozen=True)
class LineRecord:
    global_index: int
    page: int
    line_index: int
    text: str


@dataclass(frozen=True)
class Marker:
    record_index: int
    page: int
    kind: str
    label: str
    title: str | None
    parent_marker_index: int | None
    original_text: str


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_text(value: str) -> str:
    dehyphenated = re.sub(r"([A-Za-zÄÖÜäöüß])-\n([A-Za-zÄÖÜäöüß])", r"\1\2", value)
    dehyphenated = re.sub(r"([A-Za-zÄÖÜäöüß])-\s*\n\s*([A-Za-zÄÖÜäöüß])", r"\1\2", dehyphenated)
    lines = [normalize_whitespace(line) for line in dehyphenated.splitlines()]
    return "\n".join([line for line in lines if line])


def roman_to_int(value: str) -> int:
    mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    last = 0
    for char in value.upper()[::-1]:
        score = mapping.get(char, 0)
        if score < last:
            total -= score
        else:
            total += score
            last = score
    return total


def is_page_number_line(line: str) -> bool:
    return bool(PAGE_NUMBER_PATTERN.match(line))


def _find_next_content_line(records: list[LineRecord], current_index: int) -> LineRecord | None:
    next_index = current_index + 1
    while next_index < len(records):
        candidate = records[next_index]
        if candidate.text:
            return candidate
        next_index += 1
    return None


def _is_probable_paragraph_heading(line_title: str | None) -> bool:
    if line_title is None:
        return True
    title = normalize_whitespace(line_title)
    if not title:
        return True

    first_char = title[0]
    if first_char.islower():
        return False

    lowered = title.lower()
    if any(hint in lowered for hint in PARAGRAPH_REFERENCE_HINTS):
        return False

    return True


def _is_probable_title_line(value: str) -> bool:
    line = normalize_whitespace(value)
    if not line:
        return False
    if is_page_number_line(line):
        return False
    if PARAGRAPH_PATTERN.match(line) or ANNEX_PATTERN.match(line):
        return False
    if ROMAN_ONLY_PATTERN.match(line) or ROMAN_INLINE_PATTERN.match(line):
        return False
    if len(line) > 140:
        return False
    if line.endswith("."):
        return False
    return line[0].isupper()


def _is_structural_marker_line(value: str) -> bool:
    line = normalize_whitespace(value)
    return bool(
        PARAGRAPH_PATTERN.match(line)
        or ANNEX_PATTERN.match(line)
        or ROMAN_ONLY_PATTERN.match(line)
        or ROMAN_INLINE_PATTERN.match(line)
    )


def _extend_hyphenated_title(records: list[LineRecord], title_line_index: int, title: str | None) -> str | None:
    if not title:
        return title

    merged = normalize_whitespace(title)
    cursor = title_line_index

    while True:
        next_index = cursor + 1
        if next_index >= len(records):
            break
        next_line = normalize_whitespace(records[next_index].text)
        if not next_line or is_page_number_line(next_line) or _is_structural_marker_line(next_line):
            break
        if merged.endswith("-"):
            merged = f"{merged[:-1]}{next_line}"
            cursor = next_index
            continue
        if merged.isupper() and next_line.isupper() and len(next_line) <= 80:
            merged = f"{merged} {next_line}"
            cursor = next_index
            continue
        break

    return merged


def _is_probable_roman_heading(line: str) -> tuple[bool, str, str | None]:
    inline = ROMAN_INLINE_PATTERN.match(line)
    if inline:
        numeral = inline.group(1).upper()
        title = normalize_whitespace(inline.group(2))
        value = roman_to_int(numeral)
        if 1 <= value <= 30 and not title.startswith("§"):
            return True, numeral, title
        return False, "", None

    only = ROMAN_ONLY_PATTERN.match(line)
    if only:
        numeral = only.group(1).upper()
        value = roman_to_int(numeral)
        if 1 <= value <= 30:
            return True, numeral, None

    return False, "", None


def extract_toc_entries(page_texts: list[str], toc_pages: set[int]) -> tuple[list[TOCEntry], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    if not toc_pages:
        flags.append(
            QualityFlag(
                code="TOC_NOT_FOUND",
                message="No table of contents page was detected.",
                severity="warning",
            )
        )
        return [], flags

    records: list[LineRecord] = []
    idx = 0
    for page_no in sorted(toc_pages):
        for line_index, raw in enumerate(page_texts[page_no - 1].splitlines()):
            cleaned = normalize_whitespace(raw)
            if not cleaned:
                continue
            records.append(
                LineRecord(
                    global_index=idx,
                    page=page_no,
                    line_index=line_index,
                    text=cleaned,
                )
            )
            idx += 1

    entries: list[TOCEntry] = []
    current_part_id: str | None = None
    current_annex_id: str | None = None

    pointer = 0
    while pointer < len(records):
        line = records[pointer].text

        if "inhaltsverzeichnis" in line.lower():
            pointer += 1
            continue

        annex_match = ANNEX_PATTERN.match(line)
        if annex_match:
            label = f"Anlage {annex_match.group(1)}"
            title = normalize_whitespace(annex_match.group(2)) or None
            title_line_index = pointer
            if title is None:
                next_rec = _find_next_content_line(records, pointer)
                if next_rec and next_rec.global_index == records[pointer].global_index + 1:
                    if _is_probable_title_line(next_rec.text):
                        title = next_rec.text
                        title_line_index = pointer + 1
            title = _extend_hyphenated_title(records, title_line_index, title)

            page_ref: int | None = None
            merged_original = line
            merged_normalized = line

            next_rec = _find_next_content_line(records, pointer)
            if next_rec and next_rec.global_index == records[pointer].global_index + 1:
                if title is None and not is_page_number_line(next_rec.text):
                    title = next_rec.text
                    merged_original = f"{line}\n{next_rec.text}"
                    merged_normalized = normalize_text(merged_original)
                    pointer += 1
                    next_rec = _find_next_content_line(records, pointer)
                if next_rec and is_page_number_line(next_rec.text):
                    page_ref = int(next_rec.text)
                    merged_original = f"{merged_original}\n{next_rec.text}"
                    merged_normalized = normalize_text(merged_original)
                    pointer += 1

            toc_id = f"toc_{len(entries) + 1:04d}"
            entry = TOCEntry(
                toc_id=toc_id,
                parent_toc_id=None,
                kind="annex",
                label=label,
                title=title,
                referenced_page=page_ref,
                original_text=merged_original,
                normalized_text=merged_normalized,
            )
            entries.append(entry)
            current_annex_id = toc_id
            current_part_id = None
            pointer += 1
            continue

        paragraph_match = PARAGRAPH_PATTERN.match(line)
        if paragraph_match:
            label = f"§ {paragraph_match.group(1)}"
            title = normalize_whitespace(paragraph_match.group(2)) or None
            title_line_index = pointer
            if title is None:
                next_rec = _find_next_content_line(records, pointer)
                if next_rec and next_rec.global_index == records[pointer].global_index + 1:
                    if _is_probable_title_line(next_rec.text):
                        title = next_rec.text
                        title_line_index = pointer + 1
            title = _extend_hyphenated_title(records, title_line_index, title)
            if not _is_probable_paragraph_heading(title):
                pointer += 1
                continue
            parent_toc_id = current_annex_id or current_part_id

            page_ref: int | None = None
            merged_original = line

            next_rec = _find_next_content_line(records, pointer)
            if next_rec and next_rec.global_index == records[pointer].global_index + 1:
                if title is None and not is_page_number_line(next_rec.text):
                    title = next_rec.text
                    merged_original = f"{line}\n{next_rec.text}"
                    pointer += 1
                    next_rec = _find_next_content_line(records, pointer)
                if next_rec and is_page_number_line(next_rec.text):
                    page_ref = int(next_rec.text)
                    merged_original = f"{merged_original}\n{next_rec.text}"
                    pointer += 1

            previous_entry = entries[-1] if entries else None
            if (
                previous_entry
                and previous_entry.kind == "paragraph"
                and previous_entry.label == label
                and previous_entry.parent_toc_id == parent_toc_id
            ):
                pointer += 1
                continue

            entries.append(
                TOCEntry(
                    toc_id=f"toc_{len(entries) + 1:04d}",
                    parent_toc_id=parent_toc_id,
                    kind="paragraph",
                    label=label,
                    title=title,
                    referenced_page=page_ref,
                    original_text=merged_original,
                    normalized_text=normalize_text(merged_original),
                )
            )
            pointer += 1
            continue

        is_roman, roman_label, inline_title = _is_probable_roman_heading(line)
        if is_roman:
            title = inline_title
            title_line_index = pointer
            merged_original = line
            next_rec = _find_next_content_line(records, pointer)
            if title is None and next_rec and next_rec.global_index == records[pointer].global_index + 1:
                if not is_page_number_line(next_rec.text):
                    title = next_rec.text
                    title_line_index = pointer + 1
                    merged_original = f"{line}\n{next_rec.text}"
                    pointer += 1
                    next_rec = _find_next_content_line(records, pointer)
            title = _extend_hyphenated_title(records, title_line_index, title)
            page_ref: int | None = None
            if next_rec and is_page_number_line(next_rec.text):
                page_ref = int(next_rec.text)
                merged_original = f"{merged_original}\n{next_rec.text}"
                pointer += 1

            entry = TOCEntry(
                toc_id=f"toc_{len(entries) + 1:04d}",
                parent_toc_id=None,
                kind="part",
                label=roman_label,
                title=title,
                referenced_page=page_ref,
                original_text=merged_original,
                normalized_text=normalize_text(merged_original),
            )
            entries.append(entry)
            current_part_id = entry.toc_id
            current_annex_id = None
            pointer += 1
            continue

        pointer += 1

    if not entries:
        flags.append(
            QualityFlag(
                code="TOC_PARSE_EMPTY",
                message="A table of contents page was detected but no entries could be parsed.",
                severity="warning",
            )
        )

    return entries, flags


def parse_sections(page_texts: list[str], toc_pages: set[int]) -> tuple[list[SectionRecord], list[QualityFlag]]:
    flags: list[QualityFlag] = []

    line_records: list[LineRecord] = []
    global_index = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        if page_number in toc_pages:
            continue
        for line_index, raw_line in enumerate(page_text.splitlines()):
            line = normalize_whitespace(raw_line)
            if not line:
                continue
            line_records.append(
                LineRecord(
                    global_index=global_index,
                    page=page_number,
                    line_index=line_index,
                    text=line,
                )
            )
            global_index += 1

    if not line_records:
        flags.append(
            QualityFlag(
                code="EMPTY_DOCUMENT",
                message="The document does not contain extractable text.",
                severity="error",
            )
        )
        return [], flags

    markers: list[Marker] = []
    current_part_marker_index: int | None = None
    current_annex_marker_index: int | None = None

    for idx, record in enumerate(line_records):
        line = record.text

        annex_match = ANNEX_PATTERN.match(line)
        if annex_match:
            label = f"Anlage {annex_match.group(1)}"
            title = normalize_whitespace(annex_match.group(2)) or None
            title_line_index = idx
            if title is None:
                next_rec = _find_next_content_line(line_records, idx)
                if next_rec and next_rec.global_index == record.global_index + 1:
                    if _is_probable_title_line(next_rec.text):
                        title = next_rec.text
                        title_line_index = idx + 1
            title = _extend_hyphenated_title(line_records, title_line_index, title)
            markers.append(
                Marker(
                    record_index=idx,
                    page=record.page,
                    kind="annex",
                    label=label,
                    title=title,
                    parent_marker_index=None,
                    original_text=line,
                )
            )
            current_annex_marker_index = len(markers) - 1
            current_part_marker_index = None
            continue

        is_roman, roman_label, inline_title = _is_probable_roman_heading(line)
        if is_roman:
            title = inline_title
            title_line_index = idx
            if title is None:
                next_rec = _find_next_content_line(line_records, idx)
                if next_rec and next_rec.global_index == record.global_index + 1:
                    if _is_probable_title_line(next_rec.text):
                        title = next_rec.text
                        title_line_index = idx + 1
            title = _extend_hyphenated_title(line_records, title_line_index, title)
            markers.append(
                Marker(
                    record_index=idx,
                    page=record.page,
                    kind="part",
                    label=roman_label,
                    title=title,
                    parent_marker_index=None,
                    original_text=line,
                )
            )
            current_part_marker_index = len(markers) - 1
            current_annex_marker_index = None
            continue

        paragraph_match = PARAGRAPH_PATTERN.match(line)
        if paragraph_match:
            label = f"§ {paragraph_match.group(1)}"
            title = normalize_whitespace(paragraph_match.group(2)) or None
            title_line_index = idx
            if title is None:
                next_rec = _find_next_content_line(line_records, idx)
                if next_rec and next_rec.global_index == record.global_index + 1:
                    if _is_probable_title_line(next_rec.text):
                        title = next_rec.text
                        title_line_index = idx + 1
            title = _extend_hyphenated_title(line_records, title_line_index, title)
            if not _is_probable_paragraph_heading(title):
                continue

            parent = current_annex_marker_index if current_annex_marker_index is not None else current_part_marker_index
            if parent is None:
                flags.append(
                    QualityFlag(
                        code="ORPHAN_PARAGRAPH",
                        message="Paragraph detected without active part/annex parent.",
                        severity="warning",
                        context={"page": record.page, "line": line},
                    )
                )
            previous_marker = markers[-1] if markers else None
            if (
                previous_marker
                and previous_marker.kind == "paragraph"
                and previous_marker.label == label
                and previous_marker.parent_marker_index == parent
                and (record.page - previous_marker.page) <= 2
            ):
                flags.append(
                    QualityFlag(
                        code="DUPLICATE_PARAGRAPH_MARKER_SKIPPED",
                        message="A duplicate paragraph marker was skipped.",
                        severity="info",
                        context={"page": record.page, "label": label},
                    )
                )
                continue

            markers.append(
                Marker(
                    record_index=idx,
                    page=record.page,
                    kind="paragraph",
                    label=label,
                    title=title,
                    parent_marker_index=parent,
                    original_text=line,
                )
            )
            continue

    if not markers:
        fallback_text = "\n".join(record.text for record in line_records)
        flags.append(
            QualityFlag(
                code="NO_MARKERS_FOUND",
                message="No structural markers were found. A fallback section was created.",
                severity="warning",
            )
        )
        return (
            [
                SectionRecord(
                    section_id="sec_0001",
                    parent_section_id=None,
                    kind="part",
                    label="Dokument",
                    title=None,
                    start_page=1,
                    end_page=line_records[-1].page,
                    original_text=fallback_text,
                    normalized_text=normalize_text(fallback_text),
                    table_ids=[],
                    flags=[],
                )
            ],
            flags,
        )

    marker_to_section_id: dict[int, str] = {}
    sections: list[SectionRecord] = []

    for marker_index, marker in enumerate(markers):
        next_marker_index = marker_index + 1
        next_start = markers[next_marker_index].record_index if next_marker_index < len(markers) else len(line_records)
        section_lines = line_records[marker.record_index:next_start]
        original_text = "\n".join(line.text for line in section_lines).strip()
        section_id = f"sec_{marker_index + 1:04d}"
        marker_to_section_id[marker_index] = section_id

        sections.append(
            SectionRecord(
                section_id=section_id,
                parent_section_id=None,
                kind=marker.kind,  # type: ignore[arg-type]
                label=marker.label,
                title=marker.title,
                start_page=section_lines[0].page,
                end_page=section_lines[-1].page,
                original_text=original_text,
                normalized_text=normalize_text(original_text),
                table_ids=[],
                flags=[],
            )
        )

    for marker_index, marker in enumerate(markers):
        if marker.parent_marker_index is None:
            continue
        parent_section_id = marker_to_section_id.get(marker.parent_marker_index)
        if parent_section_id:
            sections[marker_index].parent_section_id = parent_section_id

    return sections, flags
