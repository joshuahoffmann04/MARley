# End-to-End Evaluation Test Documentation

**Test files:** `evaluation/tests/end_to_end/test_config.py`, `evaluation/tests/end_to_end/test_evaluate.py`, `evaluation/tests/end_to_end/test_metrics.py`
**Total tests:** 40 (40 unit)
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

### test_metrics.py (13 tests)

| Class | Tests | What is verified |
|---|---|---|
| TestComputeE2EConfigMetrics | 9 | Empty results return zeros, correct abstention precision/recall/F1, all correct abstentions, no abstentions expected or made, abstention rate, Level 1/2 counts, avg confidence, per-category breakdown, config name preserved |
| TestBuildComparisonTable | 4 | Table rows match config count, sorted by abstention_f1 descending, all fields present, rounding applied |

---

## Test Approach

- Uses `StubRetriever` (keyword-overlap scoring) and `StubGenerator` (keyword-triggered abstention)
- All metrics derived automatically from `expected_abstention` flag — no human judgements required
- No external dependencies (no Ollama, no ChromaDB)
- Tests verify both Level 1 and Level 2 abstention detection in E2E context

---

## Fixtures

No shared fixtures — tests construct `StubRetriever` and `StubGenerator` inline.

---

## CI Considerations

- All 40 tests run without external dependencies.
- No integration tests included.
- FusionRetriever tests are documented separately in `docs/testing/retrieval/test_fusion.md`.
