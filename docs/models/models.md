# Data Models

**Module:** `src/marley/models/`

The models package defines shared data classes and utility functions used across all pipeline stages. It is the foundation that ensures a consistent data contract between extractor, chunker, retrieval, and evaluation.

---

## Module Structure

```
src/marley/models/
├── __init__.py       # Re-exports all public symbols
├── extraction.py     # ExtractionResult, Section, Table
├── quality.py        # QualityFlag
└── chunking.py       # compute_token_stats()
```

---

## Data Classes

### ExtractionResult

Complete output of the PDF extractor. Represents a full document decomposed into hierarchical sections.

```python
@dataclass
class ExtractionResult:
    source_file: str         # Path to the source PDF
    total_pages: int         # Number of pages in the PDF
    sections: list[Section]  # Ordered list of document sections
```

### Section

A structural unit of the StPO document (paragraph, part, appendix, etc.).

```python
@dataclass
class Section:
    section_id: str                       # Unique ID (e.g., "par-23", "appendix-2")
    label: str                            # Display label (e.g., "§23")
    title: str                            # Section title
    kind: str                             # "preamble", "toc", "part", "paragraph", "appendix"
    start_page: int                       # First page (1-indexed)
    end_page: int                         # Last page (1-indexed)
    text: str                             # Full plain text
    tables: list[Table] = []              # Tables on this section's pages
    parent_section_id: str | None = None  # Parent part ID for paragraphs
```

### Table

A table extracted from a PDF page.

```python
@dataclass
class Table:
    table_id: str          # Unique ID (e.g., "par-6-tbl-1")
    page: int              # Page where the table starts
    headers: list[str]     # Column header strings
    rows: list[list[str]]  # Data rows
```

### QualityFlag

A diagnostic flag raised during pipeline processing. Used by the extractor, both chunkers, and potentially later stages.

```python
@dataclass
class QualityFlag:
    code: str             # Machine-readable code (e.g., "SECTION_EMPTY")
    message: str          # Human-readable description
    severity: str         # "info", "warning", or "error"
    context: dict = {}    # Additional context (section_id, page, etc.)
```

---

## Utility Functions

### `compute_token_stats(token_counts)`

Computes min, median, max, and total token statistics from a list of token counts. Used by both the PDF chunker and the FAQ chunker to avoid duplicating this logic.

```python
from src.marley.models import compute_token_stats

stats = compute_token_stats([100, 200, 300])
# {"min_tokens": 100, "median_tokens": 200, "max_tokens": 300, "total_tokens": 600}
```

Returns all zeros for an empty list.

---

## Usage

```python
from src.marley.models import ExtractionResult, Section, Table, QualityFlag, compute_token_stats
```

All symbols are re-exported from `src.marley.models`, so downstream modules never need to import from submodules directly.
