# Generation Evaluation Test Documentation

**Test files:** `evaluation/tests/generation/test_metrics.py`, `evaluation/tests/generation/test_evaluate.py`, `evaluation/tests/generation/test_combined.py`
**Total tests:** 36
**Run command:** `python -m pytest evaluation/tests/generation/ -v`

Correctness assessment tests are in the [Manual Evaluation Tests](../evaluation/test_manual_evaluation.md).

---

## Test Structure

### test_metrics.py (5 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestComputeGenerationMetrics` | 5 | Empty results, single result, multiple levels, metadata fields, distractor level sorting. |

### test_evaluate.py (17 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestSelectDistractors` | 5 | Relevant chunks excluded, count limit, zero distractors, max available limit, determinism. |
| `TestAssembleContext` | 4 | Correct chunk inclusion, distractor limiting, deterministic shuffling, zero-distractor case. |
| `TestRunGenerationEvaluation` | 8 | Unanswerable skipping, answerable inclusion, all distractor levels, result type, answer recording, context growth with distractors, empty relevant_chunks skipping, progress callback. |

---

## Test Strategy

All tests use a **stub implementation** of `Generator`:

- `StubGenerator`: Returns a fixed answer for any query.

This allows full testing of the evaluation pipeline (distractor selection, context assembly, metric aggregation) without requiring a running LLM server. The stub follows the same abstract `Generator` interface used by the real implementations.

---

## Fixtures

No shared fixtures — tests construct `StubGenerator` and synthetic data inline.

---

## CI Considerations

- All 36 tests run without external dependencies.
- No integration tests are included in the evaluation test suite (integration testing is covered by `tests/generator/test_generator.py`).

### test_combined.py (14 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestRunCombinedGenerationEvaluation` | 8 | Multi-KB data merging, relevant chunk set union, distractor selection from merged corpus, unanswerable skipping, progress callback, result type. |
| `TestRunAndReportCombined` | 6 | Report structure, combination naming, config fields, metrics, result serialisation. |

See also: [Combined Generation Test Documentation](../evaluation/test_combined_generation.md).
