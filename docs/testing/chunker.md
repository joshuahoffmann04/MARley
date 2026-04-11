# Chunker Tests

> Tests for StPO PDF chunking and FAQ chunking in `src/marley/chunker/`.

**Test files**: `tests/chunker/test_pdf_chunker.py` (60 tests), `tests/chunker/test_faq_chunker.py` (35 tests)
**Total**: 95 tests

---

## test_pdf_chunker.py — StPO Sentence-Level Chunking

### Unit Tests (No Data Files Required)

#### TestSplitSentences (4 tests)

| Test | Validates |
|------|-----------|
| `test_empty_input` | Empty string returns empty list |
| `test_whitespace_only` | Whitespace-only input returns empty list |
| `test_single_sentence` | Single sentence is extracted |
| `test_multiple_sentences` | Multiple sentences are split correctly |

#### TestSplitOversizedSentence (3 tests)

| Test | Validates |
|------|-----------|
| `test_normal_sentence_unchanged` | Sentence within budget returns as-is |
| `test_oversized_splits` | Long text is split into parts within token limit |
| `test_empty_input` | Empty string returns `[""]` |

#### TestPrepareSentences (4 tests)

| Test | Validates |
|------|-----------|
| `test_empty_list` | Empty list returns empty flat list and counts |
| `test_normal_sentences` | Normal sentences pass through with correct token counts |
| `test_oversized_sentence_split` | Oversized sentence is split, all parts within limit |
| `test_counts_match_encoder` | Token counts match tiktoken encoder output |

#### TestSlidingWindowChunks (8 tests)

Core sliding-window algorithm tests.

| Test | Validates |
|------|-----------|
| `test_empty_input` | Empty input returns empty list |
| `test_single_sentence_fits` | Single sentence becomes one chunk |
| `test_all_sentences_fit_one_chunk` | All sentences fit in one chunk |
| `test_overflow_creates_multiple_chunks` | Token overflow triggers new chunks |
| `test_overlap_repeats_trailing_sentences` | Sentence-aligned overlap between consecutive chunks |
| `test_zero_overlap_no_repetition` | Zero overlap means no repeated sentences |
| `test_oversized_sentence_gets_own_chunk` | Oversized sentence is isolated in its own chunk |
| `test_forward_progress_guaranteed` | Algorithm always advances even with large overlap |

#### TestMergeUndersized (5 tests)

| Test | Validates |
|------|-----------|
| `test_no_merging_needed` | Chunks above minimum are not merged |
| `test_merge_into_next` | Undersized chunk merges forward into next |
| `test_merge_into_prev` | Trailing undersized chunk merges backward |
| `test_single_chunk` | Single undersized chunk stays as-is |
| `test_middle_undersized_merged` | Middle undersized chunk is absorbed |

#### TestBuildHeadingPrefix (4 tests)

| Test | Validates |
|------|-----------|
| `test_paragraph_with_part_parent` | Paragraph gets hierarchical prefix (Part > Paragraph) |
| `test_top_level_section` | Top-level section gets its own label as prefix |
| `test_appendix` | Appendix gets "Appendix N" prefix |
| `test_section_without_label` | Section without label returns `None` prefix |

#### TestApplyHeadingPrefix (4 tests)

| Test | Validates |
|------|-----------|
| `test_no_heading` | `None` heading passes chunks through unchanged |
| `test_heading_prepended` | Heading is prepended to chunk text |
| `test_empty_chunks` | Empty chunk list returns empty |
| `test_body_truncated_to_budget` | Body is truncated when heading + body exceeds budget |

#### TestSerializeTableRow (2 tests)

| Test | Validates |
|------|-----------|
| `test_normal_row` | Cells joined with ` | ` separator |
| `test_row_with_empty_cells` | Empty cells are filtered out |

#### TestBuildTableChunks (3 tests)

| Test | Validates |
|------|-----------|
| `test_small_table_single_chunk` | Small table fits in one chunk with headers |
| `test_large_table_multiple_chunks` | Large table splits across chunks, each repeating headers |
| `test_empty_table` | Table with no rows produces no chunks |

#### TestChunkId (2 tests)

| Test | Validates |
|------|-----------|
| `test_text_chunk_id_format` | Text chunks get `par-1-txt-1` format ID |
| `test_table_chunk_id_format` | Table chunks get `par-1-tbl-*` format ID |

### Integration Tests (Require stpo-extracted.json)

All integration tests are guarded with `@pytest.mark.integration` and a skipif
for `stpo-extracted.json`. They run `chunk_stpo()` on the full extraction.

**Fixtures** (module-scoped): `extraction` (loads JSON), `chunking_result` (runs chunker)

#### TestChunkingBasics (3 tests)

| Test | Validates |
|------|-----------|
| `test_total_chunks_positive` | Produces at least 1 chunk |
| `test_all_chunks_have_text` | Every chunk has non-empty text |
| `test_all_chunks_have_metadata` | Every chunk has metadata with section_id |

#### TestTokenBounds (3 tests)

| Test | Validates |
|------|-----------|
| `test_no_chunk_exceeds_max` | No chunk exceeds 512 tokens |
| `test_stats_match_chunks` | Stats (total, text, table counts) match actual chunks |
| `test_token_stats_consistent` | min/max/total token stats are correct |

#### TestSlidingWindowOverlap (1 test)

| Test | Validates |
|------|-----------|
| `test_multi_chunk_sections_have_overlap` | Sections with 3+ text chunks show sentence-aligned overlap |

#### TestSectionCoverage (3 tests)

| Test | Validates |
|------|-----------|
| `test_all_sections_produce_chunks` | Every section with text or tables produces chunks |
| `test_paragraph_chunks_have_heading` | Paragraph chunks have heading path with >= 2 levels |
| `test_all_section_kinds_chunked` | paragraph, appendix, and part kinds are all represented |

#### TestTableChunking (4 tests)

| Test | Validates |
|------|-----------|
| `test_appendix_2_produces_table_chunks` | Appendix 2 generates table chunks |
| `test_table_chunks_repeat_headers` | Multi-chunk tables repeat header row |
| `test_table_chunk_ids_contain_section` | Table chunk IDs embed their section ID |
| `test_table_metadata_has_table_id` | Table chunks have `table_id` in metadata |

#### TestHeadingPaths (3 tests)

| Test | Validates |
|------|-----------|
| `test_paragraph_includes_part_in_path` | Paragraph heading path includes Part numeral |
| `test_appendix_has_correct_path` | Appendix heading path includes "Appendix" |
| `test_preamble_path` | Preamble chunks have `preamble` section kind |

#### TestQualityFlags (2 tests)

| Test | Validates |
|------|-----------|
| `test_no_error_flags` | No error-severity quality flags |
| `test_stats_match_actual` | sections_processed and tables_processed > 0 |

#### TestSaveAndLoad (2 tests)

| Test | Validates |
|------|-----------|
| `test_json_roundtrip` | Save + reload preserves stats and chunk count |
| `test_save_creates_parent_dirs` | Nested output directories are created |

---

## test_faq_chunker.py — FAQ Chunking

### Unit Tests

#### TestFormatChunkText (3 tests)

| Test | Validates |
|------|-----------|
| `test_normal_qa` | Formats as `Question: ...\nAnswer: ...` |
| `test_strips_whitespace` | Leading/trailing whitespace is stripped |
| `test_multiline_answer` | Multi-line answers are preserved |

#### TestBuildChunkId (3 tests)

| Test | Validates |
|------|-----------|
| `test_stpo_format` | StPO chunk ID: `faq-stpo-stpo-0001` |
| `test_ao_format` | AO chunk ID: `faq-ao-ao-0001` |
| `test_custom_id` | Custom IDs are handled correctly |

#### TestValidateEntry (5 tests)

| Test | Validates |
|------|-----------|
| `test_valid_entry` | Valid entry passes with no flags |
| `test_missing_id` | Empty ID returns `False` + `FAQ_ENTRY_INVALID` flag |
| `test_missing_question` | Empty question returns `False` + `FAQ_EMPTY_QUESTION` flag |
| `test_missing_answer` | Empty answer returns `False` + `FAQ_EMPTY_ANSWER` flag |
| `test_duplicate_id` | Duplicate ID returns `False` + `FAQ_ID_DUPLICATE` flag |

#### TestLoad (4 tests)

| Test | Validates |
|------|-----------|
| `test_valid_json` | Loads valid FAQ JSON with metadata and entries |
| `test_missing_entries_key` | Missing `entries` key returns empty dataset |
| `test_empty_entries` | Empty entries list returns empty dataset |
| `test_non_dict_entry_skipped` | Non-dict entries are silently skipped |

#### TestChunkFAQ (4 tests)

| Test | Validates |
|------|-----------|
| `test_single_entry` | Single entry produces one chunk with correct ID and text |
| `test_multiple_entries` | Multiple entries produce corresponding chunks |
| `test_empty_dataset` | Empty dataset produces zero chunks |
| `test_mixed_valid_invalid` | Invalid entries are skipped, valid ones chunked |

#### TestComputeStats (2 tests)

| Test | Validates |
|------|-----------|
| `test_normal_stats` | Stats correctly computed for chunk list |
| `test_empty_chunks` | Zero chunks with correct skip count |

### Integration Tests

#### TestFAQStPOChunking (5 tests, require faq-stpo.json)

| Test | Validates |
|------|-----------|
| `test_total_chunks` | Exactly 1039 chunks produced |
| `test_all_chunks_have_text` | Every chunk has non-empty text |
| `test_all_chunks_have_metadata` | All chunks have `faq_source` and `faq_id` metadata |
| `test_chunk_ids_unique` | All chunk IDs are unique |
| `test_chunk_ids_start_with_faq_stpo` | All IDs start with `faq-stpo-` |

#### TestFAQAOChunking (2 tests, require faq-ao.json)

| Test | Validates |
|------|-----------|
| `test_placeholder_produces_no_chunks` | Placeholder AO data produces zero chunks |
| `test_placeholder_raises_all_skipped` | `FAQ_ALL_SKIPPED` quality flag is raised |

#### TestChunkContent (3 tests, require faq-stpo.json)

| Test | Validates |
|------|-----------|
| `test_text_starts_with_question` | Every chunk text starts with `Question:` |
| `test_text_contains_answer` | Every chunk text contains `\nAnswer:` |
| `test_source_reference_populated` | Every chunk has a source reference |

#### TestQualityFlags (2 tests, require faq-stpo.json)

| Test | Validates |
|------|-----------|
| `test_no_error_flags` | No error-severity quality flags |
| `test_stats_match_chunks` | `entries_processed` matches chunk count |

#### TestSaveAndLoad (2 tests, require faq-stpo.json)

| Test | Validates |
|------|-----------|
| `test_json_roundtrip` | Save + reload preserves stats and chunk count |
| `test_creates_parent_dirs` | Nested directories are created |
