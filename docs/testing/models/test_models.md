# Data Models Test Documentation

**Test files:** `tests/models/test_models.py`, `tests/models/test_scoring.py`
**Total tests:** 40 (40 unit)
**Run command:** `python -m pytest tests/models/ -v`

---

## Test Strategy

Tests verify all data classes exported from `src/marley/models/` and the scoring utility functions. Each data class is tested for correct construction, default values, and serialization. The scoring module is tested for mathematical correctness of normalization formulas, edge cases, and error handling. All tests are pure unit tests with no external dependencies.

---

## Test Classes

### Data Classes — test_models.py (20 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestTable` | 2 | Construction with all fields, `asdict` serialization. |
| `TestSection` | 3 | Construction with defaults, `tables` default factory, `parent_section_id` nullable. |
| `TestExtractionResult` | 2 | Construction, `asdict` round-trip serialization. |
| `TestGenerationResult` | 2 | Construction with defaults, `context_chunk_ids` default factory. |
| `TestAbstentionResult` | 3 | Answered state, Level 1 abstention state, `retrieval_results` default factory. |
| `TestQualityFlag` | 2 | Construction, `context` default factory. |
| `TestComputeTokenStats` | 3 | Empty list returns zeros, single value, multiple values with correct median. |
| `TestSaveJson` | 3 | Saves dataclass to JSON, creates parent directories, UTF-8 encoding preserved. |

### Scoring Functions — test_scoring.py (20 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestNormalizeBM25` | 5 | Zero score maps to 0, saturation curve shape with default k, custom k parameter, empty results, order preserved. |
| `TestNormalizeVector` | 3 | Scores unchanged (identity), empty results, metadata preserved. |
| `TestNormalizeRRF` | 4 | Theoretical max maps to 1.0, partial scores scaled correctly, custom k_rrf, empty results. |
| `TestNormalizeInvalidStrategy` | 1 | Unknown strategy raises `ValueError`. |
| `TestFilterByThreshold` | 4 | Filters below threshold, keeps all above, empty after filtering, exact threshold value kept. |
| `TestComputeConfidence` | 3 | Returns max score, empty results returns 0.0, single result. |

---

## Fixtures

No shared fixtures — all tests use inline data construction.

---

## CI Considerations

- All 40 tests are pure unit tests with no external dependencies.
- No `pytest.mark.skipif` guards needed.
- Fast execution (~0.01s total).
