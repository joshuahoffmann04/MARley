# Generation Evaluation Tests

> Tests for `evaluation/generation/` — metrics, single-KB evaluation with RAGAS stubs, and combined-KB evaluation.

## RAGAS Stubbing Strategy

All generation evaluation tests patch `_score_with_ragas` to avoid Ollama/RAGAS dependencies:

```python
@patch("evaluation.generation.evaluate._score_with_ragas", fake_ragas_scores)
class TestRunGenerationEvaluation:
    ...
```

The `fake_ragas_scores` stub returns fixed scores:
- `faithfulness`: 0.9
- `answer_relevancy`: 0.85
- `factual_correctness`: 0.8

## test_metrics.py

Tests `GenerationEvalResult`, `GenerationMetrics`, and `compute_generation_metrics()`.

### TestComputeGenerationMetrics (5 tests)

| Test | Scenario |
|---|---|
| `test_empty_results` | Empty input → zeros |
| `test_single_result` | Single result counted correctly |
| `test_multiple_levels` | Multiple distractor levels grouped correctly |
| `test_metadata_fields` | KB name and model propagated |
| `test_distractor_levels_sorted` | `results_by_distractors` keys sorted |

### TestGenerationEvalResultFields (3 tests)

| Test | Scenario |
|---|---|
| `test_default_ragas_fields_are_zero` | Default RAGAS scores = 0.0 |
| `test_ragas_fields_settable` | RAGAS scores can be set explicitly |
| `test_context_chunk_ids_default_empty` | Default context_chunk_ids = [] |

### TestComputeGenerationMetricsRagasAverages (7 tests)

| Test | Scenario |
|---|---|
| `test_avg_faithfulness` | Macro-average of faithfulness scores |
| `test_avg_answer_relevance` | Macro-average of answer_relevance scores |
| `test_avg_correctness` | Macro-average of correctness scores |
| `test_all_ragas_scores_averaged` | All three RAGAS metrics averaged correctly |
| `test_empty_results_ragas_zeros` | Empty input → all averages = 0.0 |
| `test_nan_scores_excluded_from_average` | NaN values from failed RAGAS scoring excluded from averages |
| `test_all_nan_scores_return_zero` | All NaN → averages = 0.0 |

## test_evaluate.py

Tests `select_distractors()`, `_assemble_context()`, and `run_generation_evaluation()`.

### TestSelectDistractors (5 tests)

| Test | Scenario |
|---|---|
| `test_excludes_relevant_chunks` | Relevant chunk IDs excluded from distractors |
| `test_returns_up_to_requested_count` | At most `max_distractors` returned |
| `test_returns_empty_for_zero` | `max_distractors=0` → empty list |
| `test_returns_at_most_available` | Can't return more than non-relevant chunks |
| `test_deterministic` | Same inputs produce same output |

### TestAssembleContext (4 tests)

| Test | Scenario |
|---|---|
| `test_includes_relevant_and_distractors` | Both types included in context |
| `test_limits_distractors` | `num_distractors` respected |
| `test_deterministic_with_same_seed` | Same seed → same shuffle order |
| `test_zero_distractors` | Only relevant chunks in context |

### TestRunGenerationEvaluation (10 tests)

All tests use `@patch("evaluation.generation.evaluate._score_with_ragas", fake_ragas_scores)`.

| Test | Scenario |
|---|---|
| `test_skips_unanswerable` | `expected_abstention=True` questions excluded |
| `test_evaluates_answerable_questions` | Answerable questions included |
| `test_all_distractor_levels` | Correct number of results per level |
| `test_result_type` | Returns `GenerationEvalResult` instances |
| `test_records_generated_answer` | Generated answer stored correctly |
| `test_context_grows_with_distractors` | More distractors → larger context |
| `test_skips_questions_without_relevant_chunks` | No relevant chunks → skipped |
| `test_progress_callback_called` | Callback invoked per question |
| `test_ragas_scores_populated` | Stubbed RAGAS scores propagated to results |
| `test_empty_corpus_no_results` | Empty corpus → no results |

## test_combined.py

Tests `run_combined_generation_evaluation()` and `run_and_report_combined()`.

Uses a `two_kb_setup` fixture with two KBs (3 chunks each, 3 shared questions).

### TestRunCombinedGenerationEvaluation (8 tests)

| Test | Scenario |
|---|---|
| `test_result_count` | 3 questions × 2 levels = 6 results |
| `test_merges_relevant_chunks_from_both_kbs` | Question with relevant chunks in both KBs |
| `test_question_with_single_kb_relevance` | Question relevant in only one KB |
| `test_distractors_from_merged_corpus` | Distractors drawn from merged corpus |
| `test_all_distractor_levels` | Default 0-10 produces 11 × 3 = 33 results |
| `test_skips_unanswerable` | Unanswerable questions excluded |
| `test_progress_callback` | Callback invoked for each question × level |
| `test_returns_generation_eval_results` | Returns `GenerationEvalResult` instances |

### TestRunAndReportCombined (7 tests)

| Test | Scenario |
|---|---|
| `test_report_structure` | All top-level keys present |
| `test_combination_from_kb_names` | Default combination = KB names joined with `+` |
| `test_custom_combination_name` | Custom name overrides default |
| `test_config_fields` | distractor_levels, model, corpus_size, KBs |
| `test_metrics_fields` | num_results, num_queries, KB, model |
| `test_results_serialised` | Results are plain dicts (not dataclasses) |
| `test_ragas_scores_in_report` | RAGAS scores present in serialized results |
