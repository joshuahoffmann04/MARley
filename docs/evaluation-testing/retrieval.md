# Retrieval Evaluation Tests

> Tests for `evaluation/retrieval/` — metrics, single-KB evaluation, combined strategies, and RRF tuning.

## test_metrics.py

Tests the 5 retrieval metric functions and the `evaluate_retriever()` aggregation.

### TestPrecisionAtK (6 tests)

| Test | Scenario |
|---|---|
| `test_all_relevant` | All retrieved chunks are relevant → 1.0 |
| `test_none_relevant` | No retrieved chunks are relevant → 0.0 |
| `test_partial` | 2 of 5 relevant → 0.4 |
| `test_k_larger_than_retrieved` | k exceeds retrieved list length |
| `test_k_zero` | k=0 → 0.0 |
| `test_k_limits_results` | Only top-k considered |

### TestRecallAtK (5 tests)

| Test | Scenario |
|---|---|
| `test_all_recalled` | All relevant chunks retrieved → 1.0 |
| `test_none_recalled` | No relevant chunks retrieved → 0.0 |
| `test_partial` | 1 of 3 relevant retrieved → 0.33 |
| `test_empty_relevant` | No relevant chunks exist → 0.0 |
| `test_k_limits_results` | Relevant chunk beyond k not counted |

### TestMRR (6 tests)

| Test | Scenario |
|---|---|
| `test_first_is_relevant` | First result relevant → 1.0 |
| `test_second_is_relevant` | Second result relevant → 0.5 |
| `test_third_is_relevant` | Third result relevant → 0.33 |
| `test_none_relevant` | No relevant results → 0.0 |
| `test_multiple_relevant_returns_first` | Multiple relevant, takes first |
| `test_empty_retrieved` | Empty result list → 0.0 |

### TestF1AtK (6 tests)

| Test | Scenario |
|---|---|
| `test_perfect` | P@k=1.0, R@k=1.0 → F1=1.0 |
| `test_zero` | P@k=0, R@k=0 → F1=0.0 |
| `test_partial` | P@3=1/3, R@3=1/2 → F1=2/5 |
| `test_empty_relevant` | No relevant → F1=0.0 |
| `test_k_zero` | k=0 → F1=0.0 |
| `test_high_precision_low_recall` | P@1=1.0, R@1=0.25 → F1=0.4 |

### TestJaccardAtK (6 tests)

| Test | Scenario |
|---|---|
| `test_perfect_overlap` | Retrieved = relevant → 1.0 |
| `test_no_overlap` | No intersection → 0.0 |
| `test_partial_overlap` | 1 of 4 union → 0.25 |
| `test_retrieved_subset_of_relevant` | 1 of 3 → 0.33 |
| `test_empty_both` | Both empty → 0.0 |
| `test_k_limits_retrieved` | k limits the retrieved set |

### TestEvaluateRetriever (4 tests)

| Test | Scenario |
|---|---|
| `test_perfect_retrieval` | All metrics = 1.0 for perfect retrieval |
| `test_no_results` | Empty input → all metrics = 0.0 |
| `test_mixed_results` | Macro-averaged mixed scenario |
| `test_returns_dataclass` | Returns `RetrievalMetrics` instance |

## test_evaluate.py

Tests the single-KB evaluation runner (`run_evaluation()`, `run_and_report()`).

Uses a `_StubRetriever` that returns predefined chunk IDs per query.

### TestLoadEvaluation (4 tests)

Tests `load_evaluation()` — loading questions from JSON evaluation files.

### TestRunEvaluation (6 tests)

| Test | Scenario |
|---|---|
| `test_skips_unanswerable` | Questions with `expected_abstention=True` are skipped |
| `test_skips_empty_relevant_chunks` | Questions with no relevant chunks skipped |
| `test_perfect_retrieval` | Perfect hits → recall=1.0, MRR=1.0 |
| `test_no_hits` | No matches → all metrics 0.0 |
| `test_returns_retrieval_metrics` | Returns `RetrievalMetrics` instance |
| `test_skip_unanswerable_false` | `skip_unanswerable=False` includes unanswerable |

### TestRunAndReport (3 tests)

Tests report dict structure: `eval_file`, `metrics`, `config`, `corpus_size`.

## test_combined.py

Tests merged pool and fusion evaluation strategies.

Uses a `_StubRetriever` that returns the first k indexed chunks (simulates simple indexing).

### TestMergeChunks (6 tests)

Tests `merge_chunks()` — merging chunk files with duplicate detection.

### TestMergeEvaluationData (7 tests)

Tests `merge_evaluation_data()` — merging `relevant_chunks` via set union across KBs.

### TestRunMergedPoolEvaluation (6 tests)

| Test | Scenario |
|---|---|
| `test_returns_report_dict` | Report has `strategy`, `metrics`, `config` |
| `test_correct_corpus_size` | Merged corpus size matches sum |
| `test_combination_field` | KB names joined with `+` |
| `test_skips_unanswerable` | Unanswerable questions excluded |
| `test_metrics_are_valid` | Metrics in valid range [0, 1] |
| `test_zero_chunk_kb_included` | Empty KB doesn't break evaluation |

### TestRunFusionEvaluation (7 tests)

| Test | Scenario |
|---|---|
| `test_returns_report_dict` | Report has `strategy=fusion`, includes `k_rrf` |
| `test_correct_corpus_size` | Total chunks across all KBs |
| `test_combination_field` | KB names joined with `+` |
| `test_skips_unanswerable` | Unanswerable questions excluded |
| `test_metrics_are_valid` | Metrics in valid range [0, 1] |
| `test_retriever_type_in_config` | Factory-produced retriever type recorded |
| `test_zero_chunk_kb_included` | Empty KB doesn't break fusion |

## test_rrf_tuning.py

Tests the k_rrf parameter sweep for Hybrid and Fusion strategies.

Uses `KeywordRetriever` from `tests/conftest.py` for realistic retrieval behavior.

### TestSweepHybridKRRF (5 tests)

| Test | Scenario |
|---|---|
| `test_returns_valid_structure` | Has `best_k_rrf`, `best_metrics`, `sweep_results`, `config` |
| `test_sweep_covers_all_values` | One result per sweep value |
| `test_best_k_rrf_in_sweep_range` | Best k_rrf is from the sweep values |
| `test_metrics_valid` | Metrics in valid range |
| `test_default_sweep_values_used` | Uses `DEFAULT_SWEEP_VALUES` when none specified |

### TestSweepFusionKRRF (5 tests)

| Test | Scenario |
|---|---|
| `test_returns_valid_structure` | Has `best_k_rrf`, `sweep_results`, `sweep_type=fusion` |
| `test_sweep_covers_all_values` | One result per sweep value |
| `test_best_k_rrf_in_sweep_range` | Best k_rrf is from the sweep values |
| `test_config_records_knowledge_bases` | KB names and total chunks in config |
| `test_edge_case_k_rrf_one` | k_rrf=1 edge case produces valid metrics |
