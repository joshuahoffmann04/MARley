# Extractor Tests

> Tests for the StPO PDF extractor in `src/marley/extractor/`.

**Test file**: `tests/extractor/test_extractor.py`
**Total**: 81 tests (unit + integration)

## Unit Tests (No PDF Required)

### TestStripPageNumber (3 tests)

| Test | Validates |
|------|-----------|
| `test_page_1_unchanged` | Page 1 text is not modified |
| `test_strips_leading_number` | Leading page number is removed from raw text |
| `test_no_number_unchanged` | Text without leading page number passes through |

### TestNormalizeWhitespace (2 tests)

| Test | Validates |
|------|-----------|
| `test_collapses_blank_lines` | Multiple blank lines collapse to one |
| `test_strips_trailing_ws` | Trailing whitespace on lines is removed |

### TestNormalizeUnicode (11 tests)

Tests the replacement of typographic Unicode characters with ASCII equivalents.

| Test | Validates |
|------|-----------|
| `test_left_double_quote` | `\u201c` -> `"` |
| `test_right_single_quote_apostrophe` | `\u2019` -> `'` |
| `test_left_single_quote` | `\u2018` -> `'` |
| `test_en_dash` | `\u2013` -> `-` |
| `test_em_dash` | `\u2014` -> `-` |
| `test_ellipsis` | `\u2026` -> `...` |
| `test_non_breaking_space` | `\u00a0` -> ` ` |
| `test_no_special_chars_unchanged` | Plain ASCII passes through |
| `test_mixed_replacements` | Multiple replacements in one string |
| `test_empty_string` | Empty string returns empty |
| `test_german_chars_preserved` | German umlauts (o, u, a) are not modified |

### TestCellText (3 tests)

| Test | Validates |
|------|-----------|
| `test_none_returns_empty` | `None` cell value returns `""` |
| `test_strips_and_replaces_newlines` | Strips whitespace, replaces newlines with spaces |
| `test_normalizes_unicode` | Applies unicode normalization to cell content |

### TestIsHeaderRow (2 tests)

| Test | Validates |
|------|-----------|
| `test_detects_header` | Row with "Name of module" is detected as header |
| `test_normal_row` | Module data row is not a header |

### TestIsSectionLabelRow (5 tests)

| Test | Validates |
|------|-----------|
| `test_section_label_detected` | "Compulsory Elective Modules" is a section label |
| `test_module_row_not_label` | Module row with LP is not a label |
| `test_short_text_not_label` | Short text is not a label |
| `test_multiple_non_empty_not_label` | Row with multiple non-empty cells is not a label |
| `test_continuation_in_later_column_not_label` | Cross-page continuation is not a label |

### TestIsContinuationRow (3 tests)

| Test | Validates |
|------|-----------|
| `test_with_lp_not_continuation` | Row with LP value is not a continuation |
| `test_no_lp_with_text_is_continuation` | Row without LP but with text is a continuation |
| `test_all_empty_not_continuation` | All-empty row is not a continuation |

### TestMergeContinuation (1 test)

| Test | Validates |
|------|-----------|
| `test_appends_text` | Continuation row text is appended to parent row cells |

### TestMergeAppendix2Continuations (1 test)

| Test | Validates |
|------|-----------|
| `test_merges_correctly` | Multi-row continuation merging produces correct output |

### TestMakeSectionId (3 tests)

| Test | Validates |
|------|-----------|
| `test_preamble` | Preamble marker -> `"preamble"` |
| `test_paragraph` | `"SS23"` marker -> `"par-23"` |
| `test_appendix` | `"Appendix 2"` marker -> `"appendix-2"` |

### TestAssignParents (7 tests)

| Test | Validates |
|------|-----------|
| `test_paragraph_gets_part_parent` | Paragraph is assigned to preceding part |
| `test_part_has_no_parent` | Parts have no parent |
| `test_preamble_has_no_parent` | Preamble has no parent |
| `test_toc_has_no_parent` | Table of contents has no parent |
| `test_appendix_has_no_parent` | Appendices have no parent |
| `test_part_switch_updates_parent` | New part resets current parent for subsequent paragraphs |
| `test_empty_list` | Empty section list does not raise |

### TestExtractErrors (1 test)

| Test | Validates |
|------|-----------|
| `test_missing_file_raises` | `FileNotFoundError` for nonexistent PDF |

## Integration Tests (Require StPO PDF)

All integration tests are guarded with `@pytest.mark.integration` and
`@pytest.mark.skipif(not PDF_PATH.exists())`. They run the full extraction
pipeline against `data/raw/msc-computer-science.pdf`.

### Fixtures (module-scoped)

| Fixture | Provides |
|---------|----------|
| `result` | `ExtractionResult` from `extract(PDF_PATH)` |
| `sections` | `result.sections` list |
| `section_map` | Dict mapping `section_id` -> `Section` |

### TestExtractionBasics (3 tests)

| Test | Validates |
|------|-----------|
| `test_total_pages` | PDF has 47 pages |
| `test_source_file` | Source file path contains expected filename |
| `test_section_count` | Extraction produces exactly 48 sections |

### TestSectionDetection (7 tests)

| Test | Validates |
|------|-----------|
| `test_preamble_exists` | Preamble detected at page 1 |
| `test_toc_exists` | Table of contents detected |
| `test_all_four_parts` | Parts I-IV all detected |
| `test_all_paragraphs_1_to_38` | All 38 paragraphs detected |
| `test_section_kinds` | Exactly 5 kinds: preamble, toc, part, paragraph, appendix |
| `test_paragraph_36_detected` | Specific edge case paragraph detected |
| `test_no_duplicate_sections` | All section IDs are unique |

### TestSectionContent (3 tests)

| Test | Validates |
|------|-----------|
| `test_preamble_has_text` | Preamble text is substantial (>50 chars) |
| `test_paragraph_text_not_empty` | All 38 paragraphs have non-empty text |
| `test_paragraph_23_mentions_thesis` | Par. 23 mentions "thesis" or "master" |

### TestParentAssignment (8 tests)

| Test | Validates |
|------|-----------|
| `test_par_1_to_3_parent_is_part_I` | Paragraphs 1-3 belong to Part I |
| `test_par_4_to_15_parent_is_part_II` | Paragraphs 4-15 belong to Part II |
| `test_par_16_to_36_parent_is_part_III` | Paragraphs 16-36 belong to Part III |
| `test_par_37_38_parent_is_part_IV` | Paragraphs 37-38 belong to Part IV |
| `test_parts_have_no_parent` | Parts I-IV have no parent |
| `test_appendices_have_no_parent` | Appendices 1-4 have no parent |
| `test_preamble_and_toc_have_no_parent` | Preamble and TOC have no parent |
| `test_save_includes_parent_section_id` | Saved JSON preserves parent IDs |

### TestPageRanges (2 tests)

| Test | Validates |
|------|-----------|
| `test_sections_cover_all_pages` | All 47 pages are covered by at least one section |
| `test_appendix_2_spans_many_pages` | Appendix 2 spans >= 10 pages |

### TestTableExtraction (12 tests)

| Test | Validates |
|------|-----------|
| `test_total_tables` | At least 20 tables extracted |
| `test_appendix_2_has_one_table` | Appendix 2 has exactly 1 consolidated table |
| `test_appendix_2_table_headers` | 7 headers, correct first two |
| `test_appendix_2_module_count` | 54 module rows |
| `test_appendix_2_cs_module_count` | 46 CS modules |
| `test_appendix_2_conditional_count` | 8 Conditional modules |
| `test_appendix_2_no_empty_rows` | No empty name/LP cells |
| `test_appendix_2_lp_numeric` | All LP values are numeric |
| `test_appendix_3_has_tables` | Appendix 3 has >= 10 tables |
| `test_appendix_4_has_tables` | Appendix 4 has >= 1 table |
| `test_table_ids_unique` | All table IDs are unique |
| `test_table_ids_contain_section_id` | Table IDs embed their parent section ID |

### TestUnicodeNormalizationIntegration (2 tests)

| Test | Validates |
|------|-----------|
| `test_no_typographic_chars_in_section_text` | No typographic Unicode in any section text |
| `test_no_typographic_chars_in_tables` | No typographic Unicode in any table cell or header |

### TestSaveAndLoad (2 tests)

| Test | Validates |
|------|-----------|
| `test_save_roundtrip` | Save + load preserves page count and section count |
| `test_save_creates_parent_dirs` | Nested output directories are created |
