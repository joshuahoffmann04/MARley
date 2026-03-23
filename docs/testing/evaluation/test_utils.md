# Evaluation Utilities Test Documentation

**Test file:** `evaluation/tests/test_utils.py`
**Total tests:** 16 (16 unit)
**Run command:** `python -m pytest evaluation/tests/test_utils.py -v`

---

## Test Strategy

Tests verify the shared utility functions in `evaluation/utils.py` that are used across all evaluation modules. Each function is tested for correct behavior, edge cases, and error handling. All tests use temporary files and inline data — no real evaluation datasets are required.

---

## Test Classes

### JSON Loading (2 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestLoadJson` | 2 | Valid JSON file loaded correctly, missing file raises exception. |

### Evaluation Loading (3 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestLoadEvaluation` | 3 | Returns `questions` list from JSON, validates structure, handles empty questions list. |

### Chunk Merging (4 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestMergeChunks` | 4 | Single file loaded, two files concatenated, duplicate `chunk_id` raises `ValueError`, empty file handled. |

### Evaluation Data Merging (3 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestMergeEvaluationData` | 3 | Single KB passthrough, two KBs merge `relevant_chunks` via set union, question in one KB only included. |

### Abstention Metrics (4 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestComputeAbstentionMetrics` | 4 | Perfect classification (precision=1, recall=1), all false positives (precision=0), empty input defaults, mixed results with correct F1 computation. |

---

## Fixtures

No shared fixtures — all tests create temporary files via `tmp_path`.

---

## CI Considerations

- All 16 tests are pure unit tests with no external dependencies.
- No `pytest.mark.skipif` guards needed.
- Fast execution (~0.01s total).
