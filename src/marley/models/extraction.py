"""Data classes for PDF extraction results.

Defines the structural representation of a PDF document decomposed
into labelled sections, each containing plain text and optional tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Table:
    """A table extracted from a PDF page.

    Attributes:
        table_id: Unique identifier (e.g. ``"par-5-tbl-1"``).
        page: 1-indexed page number where the table appears.
        headers: Column header strings.
        rows: Data rows, each a list of cell strings.
    """

    table_id: str
    page: int
    headers: list[str]
    rows: list[list[str]]


@dataclass
class Section:
    """A structural section of a PDF document.

    Attributes:
        section_id: Unique identifier (e.g. ``"par-5"``, ``"appendix-2"``).
        label: Display label (e.g. ``"§5"``, ``"Appendix 2"``).
        title: Section title text.
        kind: One of ``"preamble"``, ``"toc"``, ``"part"``,
            ``"paragraph"``, ``"appendix"``.
        start_page: First page (1-indexed) of this section.
        end_page: Last page (1-indexed) of this section.
        text: Full extracted text content.
        tables: Tables contained within this section.
        parent_section_id: ID of the parent section, if any.
    """

    section_id: str
    label: str
    title: str
    kind: str
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
