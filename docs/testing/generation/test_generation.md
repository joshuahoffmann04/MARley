# Generation Evaluation Test Documentation

**Test files:** `evaluation/tests/generation/`
**Total tests:** 58
**Run command:** `python -m pytest evaluation/tests/generation/ -v`

---

## Test Structure

### test_metrics.py (11 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestComputeGenerationMetrics` | 5 | Empty results, single result, multiple levels, metadata fields, distractor level sorting. |
| `TestGenerationEvalResultQualityFields` | 3 | Default quality fields are zero, all quality fields are settable. |
| `TestComputeGenerationMetricsQuality` | 4 | ROUGE averaging, BERTScore averaging, judge score averaging, empty results yield zero quality scores. |

### test_evaluate.py (22 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestSelectDistractors` | 5 | Relevant chunks excluded, count limit, zero distractors, max available limit, determinism. |
| `TestAssembleContext` | 4 | Correct chunk inclusion, distractor limiting, deterministic shuffling, zero-distractor case. |
| `TestRunGenerationEvaluation` | 13 | Unanswerable skipping, answerable inclusion, all distractor levels, result type, answer recording, context growth with distractors, empty relevant_chunks skipping, progress callback, ROUGE scores populated, BERTScore populated, judge scores zero without judge, judge scores populated with judge, empty corpus yields no results. |

### test_combined.py (14 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestRunCombinedGenerationEvaluation` | 8 | Multi-KB data merging, relevant chunk set union, distractor selection from merged corpus, unanswerable skipping, progress callback, result type. |
| `TestRunAndReportCombined` | 6 | Report structure, combination naming, config fields, metrics, result serialisation. |

### test_hf_metrics.py (11 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestComputeRouge` | 7 | Empty input, output count matches input, dict keys present, identical strings score 1.0, unrelated strings score 0.0, all scores in [0,1], partial overlap yields intermediate score. |
| `TestComputeBertscore` | 4 (integration) | Empty input, output count, identical strings score ≥0.99, scores in [0,1]. |

---

## Test Strategy

All non-integration tests use a **stub implementation** of `Generator` (`StubGenerator`)
that returns a fixed answer for any query. For judge tests, an inline `_FixedJudge` stub
is used. This allows full pipeline testing without a running LLM server.

ROUGE tests run with real computation (no model, no I/O). BERTScore integration tests
are marked `@pytest.mark.integration` and skipped in standard CI runs.

---

## Fixtures

No shared fixtures — tests construct stubs and synthetic data inline.

---

## CI Considerations

- All **54 unit tests** run without external dependencies.
- **4 integration tests** (`TestComputeBertscore`) require BERTScore model download on first run and are skipped in standard CI.

See also: [Combined Generation Test Documentation](../evaluation/test_combined_generation.md).
