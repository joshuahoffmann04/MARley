# FAQ Chunker Test Documentation

**Test file:** `tests/chunker/test_faq_chunker.py`
**Total tests:** 36
**Run command:** `python -m pytest tests/chunker/test_faq_chunker.py -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify individual chunking functions in isolation using synthetic inputs. These tests run without external data files and are fast.
2. **Integration tests** run the full chunking pipeline against the real FAQ JSON files and verify structural properties of the output. These tests are skipped automatically if the JSON files are not present (`pytest.mark.skipif`).

Integration tests share a single `FAQChunkingResult` per FAQ source via module-scoped fixtures to avoid re-chunking for every test.

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | Functions covered |
|---|---|---|
| `TestFormatChunkText` | 3 | `_format_chunk_text` |
| `TestBuildChunkId` | 3 | `_build_chunk_id` |
| `TestValidateEntry` | 5 | `_validate_entry` |
| `TestLoad` | 4 | `load` |
| `TestChunkFAQ` | 4 | `chunk_faq` |
| `TestComputeStats` | 2 | `_compute_stats` |

### Integration Tests (require FAQ JSON files)

| Class | Tests | What is verified |
|---|---|---|
| `TestFAQStPOChunking` | 5 | 999 chunks, all have text, all have metadata, IDs unique, IDs prefixed correctly. |
| `TestFAQAOChunking` | 3 | 50 chunks, all have text, IDs unique. |
| `TestChunkContent` | 3 | Text starts with "Question:", contains "Answer:", source reference populated. |
| `TestQualityFlags` | 2 | No error-level flags, stats match chunk count. |
| `TestSaveAndLoad` | 2 | JSON roundtrip preserves chunk count and stats, parent directory creation. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `stpo_dataset` | module | Loads `faq-stpo.json` into a `FAQDataset`. |
| `stpo_result` | module | Runs `chunk_faq(stpo_dataset)` once for the entire test module. |
| `ao_dataset` | module | Loads `faq-ao.json` into a `FAQDataset`. |
| `ao_result` | module | Runs `chunk_faq(ao_dataset)` once for the entire test module. |

---

## CI Considerations

- All FAQ-StPO integration tests are guarded by `pytest.mark.skipif(not FAQ_STPO_PATH.exists())`.
- All FAQ-AO integration tests are guarded by `pytest.mark.skipif(not FAQ_AO_PATH.exists())`.
- If the FAQ JSON files are not available (e.g., in CI without test data), only unit tests run (21 tests).
- FAQ JSON paths resolve to `{project_root}/data/knowledgebase/faq-stpo.json` and `faq-ao.json`.
