# Extractor Test Documentation

**Test file:** `tests/extractor/test_extractor.py`
**Total tests:** 47
**Run command:** `python -m pytest tests/extractor/ -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify individual helper functions in isolation using synthetic inputs. These tests run without the PDF and are fast.
2. **Integration tests** run the full extraction pipeline against the actual StPO PDF and verify structural properties of the output. These tests are skipped automatically if the PDF is not present (`pytest.mark.skipif`).

Integration tests share a single `ExtractionResult` via module-scoped fixtures to avoid re-extracting the PDF for every test.

---

## Test Classes

### Unit Tests (no PDF required)

| Class | Tests | Functions covered |
|---|---|---|
| `TestStripPageNumber` | 3 | `_strip_page_number` |
| `TestNormalizeWhitespace` | 2 | `_normalize_whitespace` |
| `TestCellText` | 2 | `_cell_text` |
| `TestIsHeaderRow` | 2 | `_is_header_row` |
| `TestIsContinuationRow` | 3 | `_is_continuation_row` |
| `TestMergeContinuation` | 1 | `_merge_continuation` |
| `TestMergeAppendix2Continuations` | 1 | `_merge_appendix2_continuations` |
| `TestMakeSectionId` | 3 | `_make_section_id` |

### Integration Tests (require PDF)

| Class | Tests | What is verified |
|---|---|---|
| `TestExtractionBasics` | 3 | Total page count, source file path, section count. |
| `TestSectionDetection` | 7 | Preamble, ToC, parts I–IV, all 38 paragraphs, section kinds, no duplicates. |
| `TestSectionContent` | 3 | Preamble text length, non-empty paragraph text, §23 mentions thesis. |
| `TestPageRanges` | 2 | All 47 pages covered by sections, Appendix 2 spans ≥10 pages. |
| `TestTableExtraction` | 12 | Total table count, Appendix 2 (1 table, 7 headers, 54 rows, 46 CS + 8 Conditional, no empty rows, numeric LP), Appendix 3 (≥10 tables), Appendix 4 (≥1 table), unique table IDs, table IDs contain section ID. |
| `TestSaveAndLoad` | 2 | JSON roundtrip preserves page count and section count, parent directory creation. |
| `TestExtractErrors` | 1 | `FileNotFoundError` for missing PDF. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `result` | module | Runs `extract(PDF_PATH)` once for the entire test module. |
| `sections` | module | `result.sections` list. |
| `section_map` | module | Dictionary mapping `section_id → Section` for fast lookup. |

---

## CI Considerations

- All integration tests are guarded by `pytestmark = pytest.mark.skipif(not PDF_PATH.exists())`.
- If the PDF is not available (e.g., in CI without test data), only unit tests run. This is by design: the PDF is copyrighted university material and not committed to the repository.
- The PDF path resolves to `{project_root}/data/raw/msc-computer-science.pdf`.
