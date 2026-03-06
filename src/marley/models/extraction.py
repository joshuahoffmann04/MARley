"""Data classes for PDF extraction results.

Defines the structural representation of a PDF document decomposed
into labelled sections, each containing plain text and optional tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Table:
    """A table extracted from a PDF page."""
    table_id: str
    page: int
    headers: list[str]
    rows: list[list[str]]


@dataclass
class Section:
    """A structural section of a PDF document."""
    section_id: str
    label: str
    title: str
    kind: str  # "preamble", "toc", "part", "paragraph", "appendix"
    start_page: int
    end_page: int
    text: str
    tables: list[Table] = field(default_factory=list)
    parent_section_id: str | None = None


@dataclass
class ExtractionResult:
    """Complete extraction output from a PDF document."""
    source_file: str
    total_pages: int
    sections: list[Section]
