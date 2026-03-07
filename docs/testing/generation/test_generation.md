# Generation Evaluation Test Documentation

**Test files:** `evaluation/tests/generation/test_judge.py`, `evaluation/tests/generation/test_metrics.py`, `evaluation/tests/generation/test_evaluate.py`
**Total tests:** 34
**Run command:** `python -m pytest evaluation/tests/generation/ -v`

---

## Test Structure

### test_judge.py (10 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestParseJudgement` | 6 | Valid JSON parsing (correct/incorrect), markdown code fence stripping, fallback on invalid JSON, keyword-based heuristic fallback, missing field defaults. |
| `TestOllamaJudgeUnit` | 4 | Interface implementation, return type, JSON format parameter, input fields in prompt. |

### test_metrics.py (6 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestComputeGenerationMetrics` | 6 | Empty results, all correct, mixed correctness, all incorrect, metadata fields, distractor level sorting. |

### test_evaluate.py (18 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestSelectDistractors` | 5 | Relevant chunks excluded, count limit, zero distractors, max available limit, determinism. |
| `TestAssembleContext` | 4 | Correct chunk inclusion, distractor limiting, deterministic shuffling, zero-distractor case. |
| `TestRunGenerationEvaluation` | 9 | Unanswerable skipping, answerable inclusion, all distractor levels, result type, answer recording, judge result recording, context growth with distractors, empty relevant_chunks skipping, progress callback. |

---

## Test Strategy

All tests use **stub implementations** of `Generator` and `Judge`:

- `StubGenerator`: Returns a fixed answer for any query.
- `StubJudge`: Always returns the same correctness verdict.

This allows full testing of the evaluation pipeline (distractor selection, context assembly, metric aggregation) without requiring a running LLM server. The stubs follow the same abstract interfaces (`Generator`, `Judge`) used by the real implementations.

---

## CI Considerations

- All 34 tests run without external dependencies.
- No integration tests are included in the evaluation test suite (integration testing is covered by `tests/generator/test_generator.py`).
