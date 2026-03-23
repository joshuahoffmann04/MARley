# Abstention Evaluation Test Documentation

**Test files:** `evaluation/tests/abstention/test_metrics.py`, `evaluation/tests/abstention/test_evaluate.py`
**Total tests:** 22
**Run command:** `python -m pytest evaluation/tests/abstention/ -v`

---

## Test Structure

### test_metrics.py (10 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestComputeAbstentionMetrics | 10 | Perfect abstention (P=R=1), no abstentions (R=0), all abstain (coverage=0), mixed results, F1 computation, false abstention rate, empty results, single answerable, single unanswerable, threshold stored |

### test_evaluate.py (12 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestRunLevel1Sweep | 6 | Threshold=0 no abstention, threshold=1 all abstain, multiple thresholds, unanswerable correctly identified, metrics per threshold, vector normalization strategy |
| TestRunAbstentionEvaluation | 6 | Level 1 triggers for unanswerable, Level 2 triggers for LLM abstention, normal answers pass through, confidence recorded, report structure, progress callback |

---

## Test Approach

- Uses StubRetriever (keyword-overlap scoring) and StubGenerator (keyword-triggered abstention)
- No external dependencies
- Tests verify both Level 1 and Level 2 abstention detection

---

## Fixtures

No shared fixtures — tests construct `StubRetriever` and `StubGenerator` inline.

---

## CI Considerations

- All 22 tests run without external dependencies.
- No integration tests included.
