# RRF Tuning Test Documentation

**Test file:** `evaluation/tests/retrieval/test_rrf_tuning.py`
**Total tests:** 10 (10 unit)
**Run command:** `python -m pytest evaluation/tests/retrieval/test_rrf_tuning.py -v`

---

## Test Strategy

Tests verify the RRF k-parameter sweep functions that evaluate different `k_rrf` values for both HybridRetriever (within-KB) and FusionRetriever (cross-KB) configurations. All tests use stub retrievers to avoid requiring real indices or embedding models.

---

## Test Classes

### Hybrid Sweep (5 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestSweepHybridKRRF` | 5 | Returns valid report structure, sweep covers all configured values, best `k_rrf` is within sweep range, all metrics are valid numbers, default sweep values used when none specified. |

### Fusion Sweep (5 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestSweepFusionKRRF` | 5 | Returns valid report structure, sweep covers all configured values, best `k_rrf` is within sweep range, config records knowledge base names, edge case `k_rrf=1` handled correctly. |

---

## Fixtures

No shared fixtures — tests construct stub retrievers and temporary data inline.

---

## CI Considerations

- All 10 tests are pure unit tests with no external dependencies.
- No embedding models or Ollama required.
- Fast execution (~0.01s total).
