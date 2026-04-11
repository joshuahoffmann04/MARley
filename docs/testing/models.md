# Models Tests

> Tests for shared dataclasses, scoring functions, and I/O utilities in `src/marley/models/`.

**Test files**: `tests/models/test_models.py` (34 tests), `tests/models/test_scoring.py` (20 tests)
**Total**: 54 tests

## test_models.py — Dataclasses and Utilities

### TestRetrievalResult (2 tests)

Tests for the `RetrievalResult` dataclass.

| Test | Validates |
|------|-----------|
| `test_construction` | All fields (chunk_id, text, score, metadata) are set correctly |
| `test_asdict` | `dataclasses.asdict()` produces the expected dictionary |

### TestLoadChunks (5 tests)

Tests for the `load_chunks` utility that loads chunk JSON files.

| Test | Validates |
|------|-----------|
| `test_loads_valid_file` | Loads a valid JSON file with `chunks` key |
| `test_file_not_found_raises` | Raises `FileNotFoundError` for missing files |
| `test_missing_chunks_key_raises` | Raises `KeyError` when `chunks` key is absent |
| `test_accepts_path_object` | Works with `pathlib.Path` objects |
| `test_empty_chunks_list` | Returns empty list for `{"chunks": []}` |

### TestValidateCorpus (7 tests)

Tests for the `validate_corpus` function that checks corpus structure.

| Test | Validates |
|------|-----------|
| `test_valid_corpus_passes` | Valid corpus does not raise |
| `test_empty_corpus_passes` | Empty corpus is accepted |
| `test_missing_chunk_id_raises` | Raises `ValueError` when `chunk_id` is missing |
| `test_missing_text_raises` | Raises `ValueError` when `text` is missing |
| `test_missing_metadata_raises` | Raises `ValueError` when `metadata` is missing |
| `test_multiple_missing_keys_reported` | Reports all missing keys in error |
| `test_error_reports_index` | Error message includes the failing document index |

### TestTable (2 tests)

| Test | Validates |
|------|-----------|
| `test_construction` | All fields (table_id, page, headers, rows) set correctly |
| `test_asdict` | Dictionary roundtrip works |

### TestSection (3 tests)

| Test | Validates |
|------|-----------|
| `test_construction_with_defaults` | Defaults: `tables=[]`, `parent_section_id=None` |
| `test_tables_default_factory` | Each instance gets its own `tables` list |
| `test_parent_section_id` | Optional `parent_section_id` is stored |

### TestExtractionResult (2 tests)

| Test | Validates |
|------|-----------|
| `test_construction` | source_file, total_pages, sections fields |
| `test_asdict_roundtrip` | Nested section survives `asdict()` conversion |

### TestGenerationResult (2 tests)

| Test | Validates |
|------|-----------|
| `test_construction_with_defaults` | Defaults: `context_chunk_ids=[]`, `prompt_tokens=0`, `completion_tokens=0` |
| `test_context_chunk_ids_default_factory` | Each instance gets its own list |

### TestAbstentionResult (3 tests)

| Test | Validates |
|------|-----------|
| `test_answered` | Non-abstaining result fields |
| `test_level1_abstention` | Level-1 abstention fields |
| `test_retrieval_results_default_factory` | Each instance gets its own list |

### TestQualityFlag (2 tests)

| Test | Validates |
|------|-----------|
| `test_construction` | code, message, severity fields; `context={}` default |
| `test_context_default_factory` | Each instance gets its own context dict |

### TestComputeTokenStats (3 tests)

| Test | Validates |
|------|-----------|
| `test_empty_list` | Returns all-zero stats for empty input |
| `test_single_value` | Single value: min = max = total |
| `test_multiple_values` | Correct min, median, max, total for a list |

### TestSaveJson (3 tests)

Tests for the `save_json` I/O utility.

| Test | Validates |
|------|-----------|
| `test_saves_dataclass` | Serializes a dataclass to JSON file |
| `test_creates_parent_dirs` | Creates nested parent directories |
| `test_utf8_encoding` | German characters (Prufungsordnung, umlauts) are not escaped |

**Fixtures used**: `tmp_path` (pytest built-in) for file I/O tests.

---

## test_scoring.py — Score Normalization and Filtering

### TestNormalizeBM25 (5 tests)

Tests for BM25 saturation normalization (`score / (score + k)`).

| Test | Validates |
|------|-----------|
| `test_zero_score_maps_to_zero` | Score 0.0 stays 0.0 |
| `test_saturation_curve_default_k` | Default k=1: score=1 -> 0.5, score=5 -> 5/6 |
| `test_custom_k_parameter` | Custom k=10: score=10 -> 0.5 |
| `test_empty_results` | Empty list returns empty list |
| `test_order_preserved` | Input order is maintained after normalization |

### TestNormalizeVector (3 tests)

Tests for vector identity normalization (scores unchanged).

| Test | Validates |
|------|-----------|
| `test_scores_unchanged` | Cosine similarity scores pass through unmodified |
| `test_empty_results` | Empty list returns empty list |
| `test_metadata_preserved` | Metadata and text survive normalization |

### TestNormalizeRRF (4 tests)

Tests for RRF normalization by theoretical maximum.

| Test | Validates |
|------|-----------|
| `test_theoretical_max_maps_to_one` | Max RRF score (n/(k+1)) maps to 1.0 |
| `test_partial_score` | Half of max maps to 0.5 |
| `test_custom_k_rrf` | Custom k and n_retrievers parameters |
| `test_empty_results` | Empty list returns empty list |

### TestNormalizeInvalidStrategy (1 test)

| Test | Validates |
|------|-----------|
| `test_unknown_strategy_raises` | `ValueError` for unknown normalization strategy |

### TestFilterByThreshold (4 tests)

Tests for threshold-based result filtering.

| Test | Validates |
|------|-----------|
| `test_filters_below_threshold` | Results below threshold are removed |
| `test_keeps_all_above` | Results above threshold are kept |
| `test_empty_after_filtering` | All below threshold returns empty list |
| `test_exact_threshold_kept` | Score exactly at threshold is kept (>=) |

### TestComputeConfidence (3 tests)

Tests for top-1 confidence computation.

| Test | Validates |
|------|-----------|
| `test_returns_max_score` | Returns the highest score from all results |
| `test_empty_results_returns_zero` | Returns 0.0 for empty input |
| `test_single_result` | Works correctly with a single result |
