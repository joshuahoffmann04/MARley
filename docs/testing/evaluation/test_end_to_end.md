# End-to-End Evaluation Test Documentation

**Test files:** `evaluation/tests/end_to_end/test_config.py`, `evaluation/tests/end_to_end/test_evaluate.py`, `evaluation/tests/end_to_end/test_prepare.py`, `evaluation/tests/end_to_end/test_metrics.py`
**Total tests:** 47 (47 unit)
**Run command:** `python -m pytest evaluation/tests/end_to_end/ -v`

---

## Test Structure

### test_config.py (10 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestE2EConfig | 3 | Config is frozen/hashable, fields stored correctly, equality comparison |
| TestGenerateAllConfigs | 7 | Total count is 33, 9 single configs, 12 merged pool configs, 12 fusion configs, all names unique, fusion configs always use rrf normalization, single/merged normalization matches retriever type |

### test_evaluate.py (17 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestLoadQuestions | 2 | Loads from JSON, correct field mapping |
| TestSweepThreshold | 4 | Returns best threshold, sweep covers all thresholds, threshold=0 minimal abstention, threshold=1 maximal abstention |
| TestRunE2EConfig | 8 | Answerable question gets answer, unanswerable triggers Level 1, LLM abstention triggers Level 2, confidence recorded, retrieval chunk IDs recorded, progress callback called, all questions processed, E2EResult fields correct |
| TestRunAndReport | 3 | Report structure complete, threshold from sweep used, abstention metrics included |

### test_prepare.py (8 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestPrepareE2EItems | 8 | Answered question maps correctly, Level 1 abstention display text, Level 2 abstention display text, item ID format, metadata fields complete, expected_abstention preserved, category preserved, empty results returns empty list |

### test_metrics.py (12 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestComputeE2EConfigMetrics | 8 | All correct = accuracy 1.0, mixed judgements, accuracy by category, abstention precision/recall/F1, empty judgements = zeros, correct abstention counted in accuracy, partial correctness in lenient only, judgement distribution |
| TestBuildComparisonTable | 4 | Table rows match config count, sorted by lenient accuracy descending, all fields present, rounding applied |

---

## Test Approach

- Uses StubRetriever (keyword-overlap scoring) and StubGenerator (keyword-triggered abstention)
- Consistent with Phase 4 abstention evaluation test patterns
- No external dependencies (no Ollama, no ChromaDB)
- Tests verify both Level 1 and Level 2 abstention detection in E2E context

---

## Fixtures

No shared fixtures — tests construct `StubRetriever` and `StubGenerator` inline.

---

## CI Considerations

- All 47 tests run without external dependencies.
- No integration tests included.
- FusionRetriever tests are documented separately in `docs/testing/retrieval/test_fusion.md`.
