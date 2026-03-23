# Abstention Pipeline Test Documentation

**Test file:** `tests/abstention/test_detection.py`
**Total tests:** 12 (12 unit)
**Run command:** `python -m pytest tests/abstention/ -v`

---

## Test Strategy

Tests verify the LLM output abstention detection logic (Level 2). Score normalization, threshold filtering, and confidence computation are tested in `tests/models/test_scoring.py`. The full abstention pipeline is tested in `tests/server/test_pipeline.py`.

---

## Test Classes

### Detection (12 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestDetectAbstention` | 8 | Exact prefix, case-insensitive, leading whitespace, non-abstention, empty string, no false positives on partial match, multiline, prefix without reason |
| `TestExtractAbstentionReason` | 4 | Extracts reason text, strips whitespace, non-abstention returns empty, empty reason |

---

## Fixtures

No fixtures — all tests use inline string inputs.

---

## CI Considerations

- All 12 tests are pure unit tests with no external dependencies.
- No `pytest.mark.skipif` guards needed.
- Related tests: `tests/models/test_scoring.py` (scoring functions), `tests/server/test_pipeline.py` (pipeline orchestration).
