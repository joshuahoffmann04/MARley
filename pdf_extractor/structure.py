from __future__ import annotations

import re
from dataclasses import dataclass

from pdf_extractor.models import QualityFlag, SectionRecord

ROMAN_SECTION_PATTERN = re.compile(r"^\s*([IVXLCDM]+)[\.\)]?\s+(.+?)\s*$")
ANNEX_PATTERN = re.compile(r"^\s*Anlage\s+(\d+)\s*(?:[:\-–]\s*(.*))?\s*$", re.IGNORECASE)
PARAGRAPH_PATTERN = re.compile(r"^\s*§+\s*(\d+[a-zA-Z]?)\s*(.*)$")


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


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _roman_to_int(value: str) -> int:
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


def _is_probable_roman_heading(line: str) -> tuple[bool, str, str | None]:
    match = ROMAN_SECTION_PATTERN.match(line)
    if not match:
        return False, "", None

    numeral = match.group(1).upper()
    title = _normalize_whitespace(match.group(2))
    value = _roman_to_int(numeral)

    if value <= 0 or value > 50:
        return False, "", None
    if len(title) < 2 or len(title) > 180:
        return False, "", None
    if title.startswith("§"):
        return False, "", None

    return True, numeral, title


def _extract_line_records(page_texts: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    global_index = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        for line_index, raw_line in enumerate(page_text.splitlines()):
            line = _normalize_whitespace(raw_line)
            if not line:
                continue
            records.append(
                LineRecord(
                    global_index=global_index,
                    page=page_number,
                    line_index=line_index,
                    text=line,
                )
            )
            global_index += 1
    return records


def parse_sections(page_texts: list[str]) -> tuple[list[SectionRecord], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    line_records = _extract_line_records(page_texts)

    if not line_records:
        flags.append(
            QualityFlag(
                code="EMPTY_DOCUMENT",
                message="Das Dokument enthält keinen extrahierbaren Text.",
            )
        )
        return [], flags

    markers: list[Marker] = []
    current_parent_marker_index: int | None = None

    for idx, record in enumerate(line_records):
        line = record.text

        annex_match = ANNEX_PATTERN.match(line)
        if annex_match:
            label = f"Anlage {annex_match.group(1)}"
            title = _normalize_whitespace(annex_match.group(2) or "") or None
            markers.append(
                Marker(
                    record_index=idx,
                    page=record.page,
                    kind="annex",
                    label=label,
                    title=title,
                    parent_marker_index=None,
                )
            )
            current_parent_marker_index = len(markers) - 1
            continue

        paragraph_match = PARAGRAPH_PATTERN.match(line)
        if paragraph_match:
            number = paragraph_match.group(1)
            title = _normalize_whitespace(paragraph_match.group(2) or "") or None
            marker = Marker(
                record_index=idx,
                page=record.page,
                kind="paragraph",
                label=f"§ {number}",
                title=title,
                parent_marker_index=current_parent_marker_index,
            )
            if current_parent_marker_index is None:
                flags.append(
                    QualityFlag(
                        code="ORPHAN_PARAGRAPH",
                        message="Paragraph ohne übergeordnete Abschnitts- oder Anlagenmarke gefunden.",
                        context={"page": record.page, "line": line},
                    )
                )
            markers.append(marker)
            continue

        is_roman, label, title = _is_probable_roman_heading(line)
        if is_roman:
            markers.append(
                Marker(
                    record_index=idx,
                    page=record.page,
                    kind="section",
                    label=label,
                    title=title,
                    parent_marker_index=None,
                )
            )
            current_parent_marker_index = len(markers) - 1

    if not markers:
        flags.append(
            QualityFlag(
                code="NO_STRUCTURE_MARKERS",
                message="Keine Abschnitts-, Paragraphen- oder Anlagenmarker erkannt. Es wird ein Fallback-Abschnitt erzeugt.",
            )
        )
        full_text = "\n".join(record.text for record in line_records)
        fallback = SectionRecord(
            section_id="sec_0001",
            parent_section_id=None,
            kind="section",
            label="Dokument",
            title=None,
            start_page=1,
            end_page=line_records[-1].page,
            text=full_text,
            table_ids=[],
            flags=[],
        )
        return [fallback], flags

    marker_to_section_id: dict[int, str] = {}
    sections: list[SectionRecord] = []

    for marker_index, marker in enumerate(markers):
        next_marker_index = marker_index + 1
        next_start = (
            markers[next_marker_index].record_index
            if next_marker_index < len(markers)
            else len(line_records)
        )
        section_lines = line_records[marker.record_index:next_start]
        text = "\n".join(line.text for line in section_lines).strip()
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
                text=text,
                table_ids=[],
                flags=[],
            )
        )

    for marker_index, marker in enumerate(markers):
        parent_index = marker.parent_marker_index
        if parent_index is None:
            continue
        if parent_index in marker_to_section_id:
            sections[marker_index].parent_section_id = marker_to_section_id[parent_index]

    return sections, flags

