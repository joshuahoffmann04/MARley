# PDF Chunker Test Documentation

**Test file:** `tests/chunker/test_pdf_chunker.py`
**Total tests:** 50
**Run command:** `python -m pytest tests/chunker/ -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify individual chunking functions in isolation using synthetic inputs. These tests run without external data files and are fast.
2. **Integration tests** run the full chunking pipeline against the real `stpo-extracted.json` and verify structural properties of the output. These tests are skipped automatically if the JSON file is not present (`pytest.mark.skipif`).

Integration tests share a single `ChunkingResult` via module-scoped fixtures to avoid re-chunking for every test.

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | Functions covered |
|---|---|---|
| `TestSplitSentences` | 4 | `_split_sentences` |
| `TestSplitOversizedSentence` | 3 | `_split_oversized_sentence` |
| `TestPackSentences` | 4 | `_pack_sentences` |
| `TestMergeUndersized` | 4 | `_merge_undersized` |
| `TestBuildHeadingPrefix` | 4 | `_build_heading_prefix` |
| `TestApplyHeadingAndOverlap` | 4 | `_apply_heading_and_overlap` |
| `TestSerializeTableRow` | 2 | `_serialize_table_row` |
| `TestBuildTableChunks` | 3 | `_build_table_chunks` |
| `TestChunkId` | 2 | `chunk_stpo` (ID format verification) |

### Integration Tests (require `stpo-extracted.json`)

| Class | Tests | What is verified |
|---|---|---|
| `TestChunkingBasics` | 3 | Total chunk count positive, all chunks have text, all chunks have metadata. |
| `TestTokenBounds` | 3 | No chunk exceeds max_tokens, stats match chunk counts, token stats consistent. |
| `TestSectionCoverage` | 3 | All sections produce chunks, paragraph chunks have heading path with part, all section kinds present. |
| `TestTableChunking` | 4 | Appendix 2 produces table chunks, headers repeated, table chunk IDs contain section ID, table metadata has table_id. |
| `TestHeadingPaths` | 3 | Paragraph paths include part numeral, appendix paths include "Appendix", preamble chunks have correct kind. |
| `TestQualityFlags` | 2 | No error-level flags, stats have positive sections_processed and tables_processed. |
| `TestSaveAndLoad` | 2 | JSON roundtrip preserves chunk count and stats, parent directory creation. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `extraction` | module | Loads `stpo-extracted.json` into an `ExtractionResult`. |
| `chunking_result` | module | Runs `chunk_stpo(extraction)` once for the entire test module. |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not EXTRACTED_PATH.exists())`.
- If `stpo-extracted.json` is not available (e.g., in CI without test data), only unit tests run. The JSON file depends on the copyrighted university PDF and is not committed to the repository.
- The JSON path resolves to `{project_root}/data/knowledgebase/stpo-extracted.json`.
