# Combined Generation Evaluation Test Documentation

**Test file:** `evaluation/tests/generation/test_combined.py`
**Total tests:** 14
**Run command:** `python -m pytest evaluation/tests/generation/test_combined.py -v`

---

## Test Strategy

All tests are pure unit tests using synthetic data written to temporary files via `tmp_path`. No external files, knowledge bases, or Ollama server are required. Tests verify the correctness of multi-KB data merging and the combined generation evaluation pipeline using a `StubGenerator` that returns a fixed answer for any query.

### Test Data Setup

The primary fixture (`two_kb_setup`) creates two KBs:

- **stpo:** 3 chunks (stpo-1, stpo-2, stpo-3) with 3 questions (q1, q2, q3)
- **faq:** 3 chunks (faq-1, faq-2, faq-3) with 3 questions (q1, q2, q3)

After merging, question q1 has relevant chunks from both KBs (stpo-1, faq-1), while q2 and q3 each have relevant chunks from only one KB. This tests the core scenario: multi-source context for the same question.

---

## Test Classes

### Combined Generation Runner Tests (`TestRunCombinedGenerationEvaluation`, 8 tests)

| Test | What is verified |
|---|---|
| `test_result_count` | 3 answerable questions × 2 distractor levels = 6 results. |
| `test_merges_relevant_chunks_from_both_kbs` | Question q1 receives relevant chunks from both KBs (stpo-1, faq-1) in the context. |
| `test_question_with_single_kb_relevance` | Question q2 with relevance in only one KB still receives its relevant chunk (stpo-2). |
| `test_distractors_from_merged_corpus` | With distractors, context includes non-relevant chunks from the merged cross-KB pool. |
| `test_all_distractor_levels` | Default distractor levels 0–10 produce 11 results per question (33 total). |
| `test_skips_unanswerable` | Questions with `expected_abstention=True` are excluded from evaluation. |
| `test_progress_callback` | Progress callback is invoked for each question × distractor level pair. |
| `test_returns_generation_eval_results` | All returned objects are `GenerationEvalResult` instances with correct fields. |

### Combined Report Tests (`TestRunAndReportCombined`, 6 tests)

| Test | What is verified |
|---|---|
| `test_report_structure` | Report contains all expected top-level keys: combination, eval_files, config, metrics, results. |
| `test_combination_from_kb_names` | Default combination name is sorted KB names joined with `+` (e.g., "faq+stpo"). |
| `test_custom_combination_name` | Custom `combination_name` parameter overrides the default in both report and config. |
| `test_config_fields` | Config contains distractor_levels, generator_model, corpus_size (sum of all KBs), and knowledge_bases. |
| `test_metrics_fields` | Metrics contain num_results, num_queries, knowledge_base (= combination name), and model. |
| `test_results_serialised` | Results are serialised as plain dicts (not dataclasses) with all expected fields. |

---

## Coverage

| Component | Coverage |
|---|---|
| `run_combined_generation_evaluation()` | Data merging, delegation to single-KB runner, distractor selection, unanswerable skipping, progress callback |
| `run_and_report_combined()` | Report structure, combination naming, config fields, metric aggregation, result serialisation |
| `evaluation/utils.py` (merge utilities) | Covered by existing tests in `evaluation/tests/retrieval/test_combined.py` (12 tests) |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `two_kb_setup` | function | Creates two KBs (stpo + faq) with 3 chunks and 3 questions each, written to `tmp_path`. |

---

## CI Considerations

- All 14 tests are pure unit tests with no external dependencies.
- No Ollama or real knowledge bases required.
