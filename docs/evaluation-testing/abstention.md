# Abstention Evaluation Tests

> Tests for `evaluation/abstention/` — abstention metrics and the two-level evaluation pipeline.

## test_metrics.py

Tests `AbstentionMetrics` and `compute_abstention_metrics()` from `evaluation/abstention/metrics.py`.

Note: `AbstentionMetrics` is also defined in `evaluation/utils.py` (the canonical location shared with E2E evaluation). The `evaluation/abstention/metrics.py` module re-exports it.

### TestComputeAbstentionMetrics (10 tests)

| Test | Scenario |
|---|---|
| `test_perfect_abstention` | All correct: P=1.0, R=1.0, F1=1.0 |
| `test_no_abstentions` | System never abstains: R=0.0, Coverage=1.0 |
| `test_all_abstain` | System always abstains: P=0.5, Coverage=0.0 |
| `test_mixed_results` | Typical mixed: 1 correct, 1 incorrect, 1 missing, 2 answered |
| `test_f1_computation` | Verifies F1 = 2·P·R/(P+R) |
| `test_false_abstention_rate` | FAR = incorrect / answerable |
| `test_empty_results` | No results → default metrics (P=1.0, R=1.0) |
| `test_single_answerable` | One answerable, answered → Coverage=1.0 |
| `test_single_unanswerable` | One unanswerable, abstained → P=1.0, R=1.0, Coverage=0.0 |
| `test_threshold_stored` | Threshold value stored in metrics |

## test_evaluate.py

Tests `run_level1_sweep()` and `run_abstention_evaluation()`.

Uses `KeywordRetriever(score_multiplier=0.3)` for realistic retrieval behavior and `StubGenerator` for controlled generation.

### Test Data

- **Corpus**: 3 chunks about study period, thesis, and seminar
- **Questions**: 3 questions — 2 answerable (q1, q3) and 1 unanswerable (q2: "Where can I park my bicycle?")

### TestRunLevel1Sweep (6 tests)

| Test | Scenario |
|---|---|
| `test_threshold_zero_no_abstention` | threshold=0.0 → minimal abstention |
| `test_threshold_one_all_abstain` | threshold=1.0 → all questions abstain |
| `test_multiple_thresholds` | Returns one entry per threshold |
| `test_unanswerable_correctly_identified` | No-overlap question abstains at low threshold |
| `test_metrics_per_threshold` | Each entry has precision, recall, f1, FAR, coverage |
| `test_vector_normalization_strategy` | Works with `vector` normalization |

### TestRunAbstentionEvaluation (6 tests)

| Test | Scenario |
|---|---|
| `test_level1_triggers_for_unanswerable` | No retrieval overlap → Level 1 abstention |
| `test_level2_triggers_for_llm_abstention` | `StubGenerator(abstain_keywords={"thesis"})` → Level 2 |
| `test_normal_answers_pass_through` | Answerable questions receive answers |
| `test_confidence_recorded` | Each result has confidence float |
| `test_report_structure` | Report has config, metrics, results |
| `test_progress_callback` | Callback invoked for each question |
