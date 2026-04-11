# End-to-End Evaluation Tests

> Tests for `evaluation/end_to_end/` — configuration generation, pipeline execution, metrics aggregation, and comparison.

## test_config.py

Tests the `E2EConfig` dataclass and the 33-config generation matrix.

### TestE2EConfig (3 tests)

| Test | Scenario |
|---|---|
| `test_config_is_frozen_and_hashable` | `frozen=True` prevents mutation; `hash()` works |
| `test_fields_stored_correctly` | All fields stored and retrievable |
| `test_equality_comparison` | Two identical configs are equal |

### TestGenerateAllConfigs (7 tests)

| Test | Scenario |
|---|---|
| `test_total_count_is_33` | 9 single + 12 merged + 12 fusion = 33 |
| `test_9_single_configs` | 3 KBs × 3 retriever types = 9 |
| `test_12_merged_pool_configs` | 4 combinations × 3 retriever types = 12 |
| `test_12_fusion_configs` | 4 combinations × 3 retriever types = 12 |
| `test_all_names_unique` | No duplicate config names |
| `test_fusion_configs_always_use_rrf_normalization` | Fusion → normalization = `rrf` |
| `test_single_and_merged_normalization_matches_retriever` | Non-fusion → `NORMALIZATION_MAP[retriever_type]` |

## test_evaluate.py

Tests `load_questions()`, `sweep_threshold()`, `run_e2e_config()`, and `run_and_report()`.

Uses `KeywordRetriever(score_multiplier=0.3)` for realistic retrieval and `StubGenerator` for generation.

### Test Data

- **Corpus**: 3 chunks (study period, thesis, seminar)
- **Questions**: 3 questions (2 answerable, 1 unanswerable)
- **Config**: `E2EConfig(name="test-config", retriever_type="bm25", strategy="single", ...)`

### TestLoadQuestions (2 tests)

| Test | Scenario |
|---|---|
| `test_loads_from_json` | Loads questions list from JSON file |
| `test_correct_field_mapping` | All question fields correctly mapped |

### TestSweepThreshold (4 tests)

| Test | Scenario |
|---|---|
| `test_returns_best_threshold` | Returns float in [0.0, 1.0] |
| `test_sweep_covers_all_thresholds` | One entry per threshold value |
| `test_threshold_zero_minimal_abstention` | Low threshold → minimal abstention |
| `test_threshold_one_maximal_abstention` | High threshold → some abstentions |

### TestRunE2EConfig (8 tests)

| Test | Scenario |
|---|---|
| `test_answerable_question_gets_answer` | Answerable → not abstained, has answer |
| `test_unanswerable_triggers_level1_at_high_threshold` | Unanswerable → Level 1 at threshold=1.0 |
| `test_llm_abstention_triggers_level2` | `abstain_keywords={"thesis"}` → Level 2 |
| `test_confidence_recorded` | Confidence ≥ 0.0 recorded |
| `test_retrieval_chunk_ids_recorded` | Chunk IDs from retrieval stored |
| `test_progress_callback_called` | Callback count = question count |
| `test_all_questions_processed` | Results count = question count |
| `test_e2e_result_fields_correct` | All E2EResult fields populated correctly |

### TestRunAndReport (3 tests)

| Test | Scenario |
|---|---|
| `test_report_structure_complete` | All keys: config, threshold, sweep, metrics, results |
| `test_threshold_from_sweep_used` | Best threshold from sweep values |
| `test_abstention_metrics_included` | Report has precision, recall, f1 |

## test_metrics.py

Tests `E2EConfigMetrics`, `compute_e2e_config_metrics()`, and `build_comparison_table()`.

### TestComputeE2EConfigMetrics (9 tests)

| Test | Scenario |
|---|---|
| `test_empty_results_returns_zeros` | Empty → all zeros |
| `test_correct_abstention_precision_recall` | Mixed: P=0.5, R=0.5, F1=0.5 |
| `test_all_correct_abstentions` | Perfect: P=1.0, R=1.0, F1=1.0 |
| `test_no_abstentions_expected_none_made` | No unanswerable, no abstentions → P=0.0, R=0.0 |
| `test_abstention_rate` | 2/4 abstained → rate=0.5 |
| `test_level1_level2_counts` | Level 1 and Level 2 counted separately |
| `test_avg_confidence` | Mean confidence across all results |
| `test_per_category_breakdown` | Per-category abstention metrics |
| `test_config_name_preserved` | Config name in output |

### TestBuildComparisonTable (4 tests)

| Test | Scenario |
|---|---|
| `test_table_rows_match_config_count` | One row per config |
| `test_sorted_by_abstention_f1_descending` | Highest F1 first |
| `test_all_fields_present` | All expected columns present |
| `test_rounding_applied` | Values rounded to 4 decimal places |
