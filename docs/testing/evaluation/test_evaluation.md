# Evaluation Test Documentation

**Test files:** `evaluation/tests/test_metrics.py`, `evaluation/tests/test_evaluate.py`
**Total tests:** 34
**Run command:** `python -m pytest evaluation/tests/ -v`

---

## Test Strategy

All evaluation tests are pure unit tests using synthetic data. No external files or knowledge bases are required. The tests verify the mathematical correctness of the retrieval metrics and the evaluation runner pipeline.

---

## Test Classes

### Metrics Tests (`test_metrics.py`, 21 tests)

| Class | Tests | Functions covered |
|---|---|---|
| `TestPrecisionAtK` | 6 | `precision_at_k` |
| `TestRecallAtK` | 5 | `recall_at_k` |
| `TestMRR` | 6 | `mrr` |
| `TestEvaluateRetriever` | 4 | `evaluate_retriever` |

Each metric is tested for perfect results, zero results, partial results, and edge cases (empty inputs, k=0, k larger than result list).

### Evaluation Runner Tests (`test_evaluate.py`, 13 tests)

| Class | Tests | Functions covered |
|---|---|---|
| `TestLoadEvaluation` | 4 | `load_evaluation` |
| `TestRunEvaluation` | 6 | `run_evaluation` |
| `TestRunAndReport` | 3 | `run_and_report` |

Runner tests use a `_StubRetriever` that returns predefined results, isolating the evaluation logic from any actual retrieval implementation. Tests verify:

- Correct loading and parsing of evaluation JSON files
- Unanswerable questions are skipped by default
- Questions with empty `relevant_chunks` are skipped
- Perfect and zero-hit retrieval scenarios
- The `skip_unanswerable` parameter
- Report dict structure and content

---

## CI Considerations

- All tests run without external data files.
- No `pytest.mark.skipif` guards needed.
- Tests use `tmp_path` fixtures for file I/O tests.
