from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from pdf_extractor.models import QualityFlag, SectionRecord, TOCEntry

# --- Regex Patterns ---
# Simplified and consolidated patterns
PATTERNS = {
    "roman_only": re.compile(r"^\s*([IVXLCDM]{1,8})\.\s*$", re.IGNORECASE),
    "roman_inline": re.compile(r"^\s*([IVXLCDM]{1,8})\.\s+(.+?)\s*$", re.IGNORECASE),
    "annex": re.compile(r"^\s*Anlage\s+(\d+)\s*:?\s*(.*)\s*$", re.IGNORECASE),
    "paragraph": re.compile(r"^\s*§+\s*(\d+[a-zA-Z]?)\s*(.*)$"),
    "page_number": re.compile(r"^\s*\d+\s*$"),
}

PARAGRAPH_REFERENCE_HINTS = {"abs.", "satz", "nr.", "gemäß", "anlage"}


@dataclass(frozen=True)
class LineRecord:
    global_index: int
    page: int
    line_index: int
    text: str


@dataclass
class Marker:
    record_index: int
    page: int
    kind: str  # "part", "annex", "paragraph"
    label: str
    title: str | None
    parent_marker_index: int | None
    original_text: str


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_text(value: str) -> str:
    # De-hyphenate at line breaks: "Wort-\nende" -> "Wortende"
    text = re.sub(r"([A-Za-zÄÖÜäöüß])-\n([A-Za-zÄÖÜäöüß])", r"\1\2", value)
    text = re.sub(r"([A-Za-zÄÖÜäöüß])-\s*\n\s*([A-Za-zÄÖÜäöüß])", r"\1\2", text)
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    return "\n".join([line for line in lines if line])


def roman_to_int(value: str) -> int:
    mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    value = value.upper()
    total = 0
    last = 0
    for char in value[::-1]:
        score = mapping.get(char, 0)
        if score < last:
            total -= score
        else:
            total += score
            last = score
    return total


class PDFStructureParser:
    """Parses PDF text lines into a structured hierarchy of sections."""

    def __init__(self, page_texts: list[str], toc_pages: set[int]):
        self.page_texts = page_texts
        self.toc_pages = toc_pages
        self.line_records: list[LineRecord] = []
        self._build_line_records()

    def _build_line_records(self):
        global_index = 0
        for page_number, page_text in enumerate(self.page_texts, start=1):
            if page_number in self.toc_pages:
                continue
            for line_index, raw_line in enumerate(page_text.splitlines()):
                line = normalize_whitespace(raw_line)
                if not line:
                    continue
                self.line_records.append(
                    LineRecord(
                        global_index=global_index,
                        page=page_number,
                        line_index=line_index,
                        text=line,
                    )
                )
                global_index += 1

    def _find_next_content_line(self, current_index: int) -> LineRecord | None:
        for i in range(current_index + 1, len(self.line_records)):
            return self.line_records[i]
        return None

    def _is_probable_paragraph_heading(self, title: str | None) -> bool:
        if not title:
            return True
        title = normalize_whitespace(title)
        if not title:
            return True
        if title[0].islower():
            return False
        lowered = title.lower()
        if any(hint in lowered for hint in PARAGRAPH_REFERENCE_HINTS):
            return False
        return True

    def _is_probable_title_line(self, value: str) -> bool:
        line = normalize_whitespace(value)
        if not line:
            return False
        if PATTERNS["page_number"].match(line):
            return False
        if PATTERNS["paragraph"].match(line) or PATTERNS["annex"].match(line):
            return False
        if PATTERNS["roman_only"].match(line) or PATTERNS["roman_inline"].match(line):
            return False
        if len(line) > 140:
            return False
        if line.endswith("."):
            return False
        return line[0].isupper()

    def _extend_title(self, start_index: int, initial_title: str | None) -> str | None:
        if not initial_title:
            return initial_title

        merged = normalize_whitespace(initial_title)
        current_idx = start_index

        while True:
            next_rec = self._find_next_content_line(current_idx)
            if not next_rec or next_rec.global_index != current_idx + 1:
                break
            
            next_line = next_rec.text
            if PATTERNS["page_number"].match(next_line): # Skip page numbers in title flow? No, breaks flow usually.
                 break
            
            # Stop if next line looks like a new marker
            if (PATTERNS["paragraph"].match(next_line) or 
                PATTERNS["annex"].match(next_line) or 
                PATTERNS["roman_only"].match(next_line) or 
                PATTERNS["roman_inline"].match(next_line)):
                break

            if merged.endswith("-"):
                merged = f"{merged[:-1]}{next_line}"
                current_idx = next_rec.global_index
            elif merged.isupper() and next_line.isupper() and len(next_line) <= 80:
                merged = f"{merged} {next_line}"
                current_idx = next_rec.global_index
            else:
                break
        
        return merged

    def parse(self) -> tuple[list[SectionRecord], list[QualityFlag]]:
        flags: list[QualityFlag] = []
        if not self.line_records:
            flags.append(QualityFlag(code="EMPTY_DOCUMENT", message="No extractable text found.", severity="error"))
            return [], flags

        markers: list[Marker] = []
        current_part_idx: int | None = None
        current_annex_idx: int | None = None

        for idx, record in enumerate(self.line_records):
            line = record.text
            
            # 1. Check Annex
            match = PATTERNS["annex"].match(line)
            if match:
                label = f"Anlage {match.group(1)}"
                title = normalize_whitespace(match.group(2)) or None
                title = self._extend_title(idx, title) if title else title
                # If no title inline, look ahead (simplified logic for robustness)
                if not title:
                     next_rec = self._find_next_content_line(idx)
                     if next_rec and self._is_probable_title_line(next_rec.text):
                         title = next_rec.text # Don't verify too deep, simple lookahead

                markers.append(Marker(idx, record.page, "annex", label, title, None, line))
                current_annex_idx = len(markers) - 1
                current_part_idx = None
                continue

            # 2. Check Roman Part
            match_inline = PATTERNS["roman_inline"].match(line)
            match_only = PATTERNS["roman_only"].match(line)
            if match_inline or match_only:
                numeral = (match_inline or match_only).group(1).upper()
                value = roman_to_int(numeral)
                if 1 <= value <= 30: # Sanity check for parts
                    title = normalize_whitespace(match_inline.group(2)) if match_inline else None
                    title = self._extend_title(idx, title)
                    label = numeral # Just the numeral as label
                    
                    markers.append(Marker(idx, record.page, "part", label, title, None, line))
                    current_part_idx = len(markers) - 1
                    current_annex_idx = None
                    continue

            # 3. Check Paragraph
            match = PATTERNS["paragraph"].match(line)
            if match:
                # Basic check: title plausibility
                label = f"§ {match.group(1)}"
                raw_title = normalize_whitespace(match.group(2)) or None
                
                # Check next line if title is empty
                next_rec = self._find_next_content_line(idx)
                if not raw_title and next_rec and self._is_probable_title_line(next_rec.text):
                     raw_title = next_rec.text
                
                title = self._extend_title(idx, raw_title)
                
                if not self._is_probable_paragraph_heading(title):
                    continue

                parent_idx = current_annex_idx if current_annex_idx is not None else current_part_idx
                if parent_idx is None:
                    flags.append(QualityFlag(code="ORPHAN_PARAGRAPH", message="Paragraph found without parent.", severity="warning", context={"page": record.page, "line": line}))

                # Duplicate detection check
                if markers:
                    prev = markers[-1]
                    if (prev.kind == "paragraph" and prev.label == label and 
                        prev.parent_marker_index == parent_idx and (record.page - prev.page) <= 2):
                        flags.append(QualityFlag(code="DUPLICATE_PARAGRAPH", message="Duplicate skipped.", severity="info", context={"label": label}))
                        continue

                markers.append(Marker(idx, record.page, "paragraph", label, title, parent_idx, line))
                continue

        if not markers:
            # Fallback one big section
            full_text = "\n".join(r.text for r in self.line_records)
            return [SectionRecord(
                section_id="sec_0001",
                kind="part",
                label="Dokument",
                start_page=1,
                end_page=self.line_records[-1].page,
                original_text=full_text,
                normalized_text=normalize_text(full_text)
            )], [QualityFlag(code="NO_MARKERS", message="No markers found, fallback used.", severity="warning")]

        # Convert markers to sections
        sections: list[SectionRecord] = []
        marker_id_map = {} # marker_index -> section_id

        for i, marker in enumerate(markers):
            # Determine content range
            start_rec_idx = marker.record_index
            if i + 1 < len(markers):
                end_rec_idx = markers[i+1].record_index
            else:
                end_rec_idx = len(self.line_records) # Until end

            # Slicing line records is O(N) but robust
            # line_records is flat, so slicing by index is correct?
            # self.line_records is a list, so yes, slicing works if record_index is index in list.
            # Wait, line_records is list of LineRecord.
            # Marker.record_index stores 'idx' from enumeration of self.line_records.
            section_lines = self.line_records[start_rec_idx:end_rec_idx]
            
            text_block = "\n".join(l.text for l in section_lines)
            sec_id = f"sec_{i+1:04d}"
            marker_id_map[i] = sec_id
            
            parent_id = None
            if marker.parent_marker_index is not None:
                parent_id = marker_id_map.get(marker.parent_marker_index)

            sections.append(SectionRecord(
                section_id=sec_id,
                parent_section_id=parent_id,
                kind=marker.kind,
                label=marker.label,
                title=marker.title,
                start_page=section_lines[0].page,
                end_page=section_lines[-1].page,
                original_text=text_block,
                normalized_text=normalize_text(text_block)
            ))

        return sections, flags


def extract_toc_entries(page_texts: list[str], toc_pages: set[int]) -> tuple[list[TOCEntry], list[QualityFlag]]:
    # Simplified TOC extraction - strict regex based
    flags: list[QualityFlag] = []
    if not toc_pages:
        return [], [QualityFlag(code="NO_TOC_PAGES", message="No TOC pages detected.", severity="warning")]

    entries: list[TOCEntry] = []
    
    # Iterate all TOC lines
    full_toc_text = []
    for p in sorted(toc_pages):
        full_toc_text.extend(page_texts[p-1].splitlines())
    
    # Only simple parsing for TOC - reliable TOC parsing is extremely hard without visual layout analysis
    # This keeps it simple: look for § X .... page
    for line in full_toc_text:
        line = normalize_whitespace(line)
        if not line: continue
        
        # Paragraph match
        match = PATTERNS["paragraph"].match(line)
        if match:
            label = f"§ {match.group(1)}"
            rest = match.group(2)
            # Try to extract page number from end
            page_match = re.search(r"\s+(\d+)$", rest)
            page = int(page_match.group(1)) if page_match else None
            title = rest[:page_match.start()].strip() if page_match else rest.strip()
            
            entries.append(TOCEntry(
                toc_id=f"toc_{len(entries)+1:04d}",
                kind="paragraph",
                label=label,
                title=title or None,
                referenced_page=page,
                original_text=line,
                normalized_text=normalize_text(line)
            ))
            continue
        
        # Annex match
        match = PATTERNS["annex"].match(line)
        if match:
            label = f"Anlage {match.group(1)}"
            rest = match.group(2)
            page_match = re.search(r"\s+(\d+)$", rest)
            page = int(page_match.group(1)) if page_match else None
            title = rest[:page_match.start()].strip() if page_match else rest.strip()

            entries.append(TOCEntry(
                toc_id=f"toc_{len(entries)+1:04d}",
                kind="annex",
                label=label,
                title=title or None,
                referenced_page=page,
                original_text=line,
                normalized_text=normalize_text(line)
            ))

    if not entries:
        flags.append(QualityFlag(code="TOC_EMPTY", message="No entries found in TOC pages.", severity="warning"))

    return entries, flags


def parse_sections(page_texts: list[str], toc_pages: set[int]) -> tuple[list[SectionRecord], list[QualityFlag]]:
    parser = PDFStructureParser(page_texts, toc_pages)
    return parser.parse()
