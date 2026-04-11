# PDF Extractor

**Module:** `src/marley/extractor/`
**Input:** `data/raw/msc-computer-science.pdf` (47 pages, English StPO)
**Output:** `data/knowledgebase/stpo-extracted.json`

The extractor converts the StPO PDF into a structured JSON file containing labelled sections with plain text and tables. It is the first stage of the MARley pipeline.

---

## Theoretical Background

PDF extraction is a foundational challenge in document-based question answering systems. Unlike plain text or HTML, PDF is a page-description format that encodes visual layout (character positions, font metrics, line art) rather than logical structure (headings, paragraphs, tables). Recovering the logical structure requires heuristics that interpret visual cues as semantic boundaries.

Two complementary libraries address this task in the Python ecosystem. **PyMuPDF** (`fitz`) provides fast, reading-order text extraction by iterating over text spans in their encoded order (Artifex Software, 2024). It excels at extracting continuous prose but has limited table awareness. **pdfplumber** (Singer-Vine, 2024) uses spatial analysis to detect table regions and reconstruct cell boundaries from ruled lines, making it the preferred choice for tabular content.

MARley uses both libraries in a two-pass design: PyMuPDF for text extraction (Stage 1) and pdfplumber for table extraction (Stage 5). This avoids the trade-off of choosing a single library and leverages each tool's strength. Section boundaries are detected via regex-based marker scanning on the extracted text, which is feasible because the StPO follows a highly regular structure (numbered paragraphs, Roman-numeral parts, labelled appendices).

---

## Processing Pipeline

```
PDF file
  │
  ├─ 1. Text extraction       (PyMuPDF)
  ├─ 2. Section detection     (regex-based marker scan)
  ├─ 3. Section assembly      (text slicing by marker boundaries)
  ├─ 4. Parent assignment     (document hierarchy)
  ├─ 5. Table extraction      (pdfplumber)
  ├─ 6. Table assignment      (page-range matching)
  │
  └── ExtractionResult → JSON
```

### Stage 1: Text Extraction

**Library:** PyMuPDF (`fitz`)
**Function:** `extract_page_texts(pdf_path) → list[tuple[int, str]]`

For each page in the PDF:

1. Extract raw text via `page.get_text("text")`, which returns text in reading order (left-to-right, top-to-bottom).
2. **Strip page numbers:** Pages 2–47 have a standalone page number as the first line. The function `_strip_page_number` removes this line if it matches the expected page number. Page 1 has no page number and is left unchanged.
3. **Normalize whitespace:** The function `_normalize_whitespace` collapses runs of multiple blank lines into single blank lines and strips trailing whitespace from each line.
4. **Normalize Unicode:** The function `_normalize_unicode` replaces typographic characters inherited from the PDF with their ASCII equivalents. This includes smart quotes (`"` `"` → `"`), curly apostrophes (`'` `'` → `'`), en/em dashes (`–` `—` → `-`), horizontal ellipsis (`…` → `...`), and non-breaking spaces. This normalization improves downstream search consistency while preserving readability.

The result is a list of `(page_number, cleaned_text)` tuples, one per page.

### Stage 2: Section Detection

**Function:** `_detect_markers(pages) → list[_Marker]`

Scans all page texts line by line using regex patterns to identify section boundaries. Each detected boundary produces a `_Marker` with: kind, label, title, page number, and line index.

**Detected marker types:**

| Pattern | Kind | Example match |
|---|---|---|
| Page 1 | `preamble` | Title page |
| Line contains "table of contents" | `toc` | Table of Contents page |
| `^\s*([IVXLC]+)\.\s*$` | `part` | `I.`, `II.`, `III.`, `IV.` |
| `^\s*§\s*(\d+[a-z]?)\s*$` | `paragraph` | `§ 1`, `§ 23`, `§ 36` |
| `^\s*Appendix\s+(\d+)\s*:\s*(.*)` | `appendix` | `Appendix 1: Example degree program curriculum` |

**Title resolution:** For paragraphs, parts, and appendices without an inline title, the extractor takes the next non-empty, non-page-number line as the title (`_next_non_empty_line`).

**Deduplication:** Some markers appear on consecutive pages (e.g., Appendix 1 header on pages 17 and 18). The function keeps only the first occurrence of each `(kind, label)` pair.

**Result:** 48 markers for this document (1 preamble + 1 ToC + 4 parts + 38 paragraphs + 4 appendices).

### Stage 3: Section Assembly

**Function:** `_build_sections(pages, markers) → list[Section]`

Converts markers into `Section` objects by computing page ranges and extracting the corresponding text:

1. **Page range:** Each section starts at its marker's page and ends at the page before the next marker (or the last page for the final section). If two markers share a page, both use that page as start and end.
2. **Text slicing:** For the first page of a section, text begins at the marker's line index. For the last page (if shared with the next section), text is cut at the next marker's line index.
3. **Section IDs** are generated by `_make_section_id`:
   - `preamble` → `"preamble"`
   - `toc` → `"toc"`
   - `I.` → `"part-I"`
   - `§23` → `"par-23"`
   - `Appendix 2` → `"appendix-2"`

### Stage 4: Parent Assignment

**Function:** `_assign_parents(sections)`

Establishes the hierarchical relationship between sections. The StPO document is organized into four numbered parts, each containing a range of paragraphs:

| Part | Paragraphs | Topic |
|---|---|---|
| `part-I` | §1–§3 | General |
| `part-II` | §4–§15 | Program-related rules |
| `part-III` | §16–§36 | Examination-related provisions |
| `part-IV` | §37–§38 | Final provisions |

The function walks through the section list in document order and tracks the most recently encountered part. Each paragraph receives the current part as its `parent_section_id`. All other section kinds — preamble, table of contents, parts themselves, and appendices — are top-level sections with `parent_section_id` set to `None`.

### Stage 5: Table Extraction

**Library:** pdfplumber
**Function:** `_extract_all_tables(pdf_path) → list[Table]`

Opens the PDF with pdfplumber and processes each page's tables independently. Tables are classified by column count:

#### Appendix 2 Tables (13 columns)

The module list in Appendix 2 spans 17 pages (pages 20–36). pdfplumber detects 13 columns per page, but only 7 contain data. The extractor:

1. **Collects rows** from all 13-column tables across all pages via `_extract_appendix2_rows`.
2. **Maps columns:** Indices `[0, 3, 4, 7, 8, 9, 10]` are mapped to the 7 canonical columns:
   - Name of module / German translation
   - LP (credit points)
   - Degree of obligation
   - Level
   - Qualification goals
   - Prerequisites
   - Prerequisites to earn credits (LP)
3. **Filters** header rows (detected by `_is_header_row`: rows containing "Name of module"), section label rows (detected by `_is_section_label_row`: single non-empty cell in column 0 with >10 characters that is not a module code; rows with text only in later columns are treated as continuation rows), and fully empty rows.
4. **Merges continuation rows** via `_merge_appendix2_continuations`. pdfplumber splits multi-line cells across rows at page boundaries. A continuation row is identified by `_is_continuation_row`: it has text content but an empty LP column (column 1). The function `_merge_continuation` appends each non-empty cell of the continuation row to the corresponding cell of the previous row. This merging happens after all pages are collected, which correctly handles cross-page boundaries.

**Result:** One merged table with 54 data rows (46 CS modules + 8 Conditional modules), 7 columns.

#### Generic Tables (non-13-column)

All other tables are processed by `_process_generic_table`:

1. Clean each cell with `_cell_text` (replace newlines with spaces, strip whitespace, normalize Unicode).
2. Remove fully empty rows.
3. Remove columns that are empty across all rows.
4. First remaining row becomes headers, rest becomes data rows.

### Stage 6: Table Assignment

**Function:** `_assign_tables(sections, tables)`

Each table is assigned to the section whose page range contains the table's start page. Tables are matched in reverse section order so that later sections take priority when pages overlap. The table ID is updated to include the section ID: `{section_id}-tbl-{n}`.

---

## Public API

```python
from src.marley.extractor import extract, save

result = extract("data/raw/msc-computer-science.pdf")
save(result, "data/knowledgebase/stpo-extracted.json")
```

| Function | Signature | Description |
|---|---|---|
| `extract` | `(pdf_path: str \| Path) → ExtractionResult` | Run the full extraction pipeline. Raises `FileNotFoundError` if the PDF does not exist. |
| `save` | `(result: ExtractionResult, output_path: str \| Path) → Path` | Serialize to JSON. Creates parent directories if needed. Returns the resolved output path. |

---

## Data Classes

The extractor produces `ExtractionResult`, `Section`, and `Table` objects defined in `src/marley/models/`. See [models.md](models.md) for details.

---

## Dependencies

| Library | Purpose |
|---|---|
| PyMuPDF (`fitz`) | Text extraction via `page.get_text("text")` |
| pdfplumber | Table detection and extraction via `page.find_tables()` |

Both libraries are pure Python or ship pre-built binaries and require no external system dependencies on Windows.

---

## Known Characteristics

- The PDF uses `§ 36` (with a space between § and number). The regex `^\s*§\s*(\d+[a-z]?)\s*$` handles this.
- Appendix 1 appears on both pages 17 and 18. The marker deduplication keeps only the first occurrence.
- Appendix 3 contains 14 separate tables (one per profile area or source program group) because pdfplumber detects separate table regions on each page.
- Appendix 4 is split into 3 tables across 3 pages.
- Sections referencing the "Allgemeine Bestimmungen" (General Regulations) contain minimal text, typically a single sentence like "The rules under §X of the General Regulations apply."
- The source PDF contains typographic characters (smart quotes, curly apostrophes, en/em dashes). These are normalized to ASCII equivalents during extraction to ensure consistent search behavior.
- The `\n` characters in JSON text fields represent actual line breaks from the PDF source. They mark paragraph boundaries, subsection starts, and list items within the original document. These are intentional and not extraction artifacts.
