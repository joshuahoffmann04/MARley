# Utility Tests

> Tests for `evaluation/utils.py` — shared functions used across all evaluation modules.

## test_utils.py

Tests the shared utility functions: `load_json()`, `load_evaluation()`, `merge_chunks()`, `merge_evaluation_data()`, and `compute_abstention_metrics()`.

### TestLoadJson (2 tests)

| Test | Scenario |
|---|---|
| `test_valid_json` | Loads and returns dict from JSON file |
| `test_missing_file_raises` | `FileNotFoundError` for missing file |

### TestLoadEvaluation (3 tests)

| Test | Scenario |
|---|---|
| `test_returns_questions_list` | Returns list of question dicts |
| `test_validates_structure` | Preserves category and expected_abstention fields |
| `test_empty_questions` | Empty questions list → empty result |

### TestMergeChunks (4 tests)

| Test | Scenario |
|---|---|
| `test_single_file` | Single file returns its chunks |
| `test_two_files_concatenated` | Two files concatenated in order |
| `test_duplicate_chunk_id_raises` | `ValueError` on duplicate `chunk_id` |
| `test_empty_file` | Empty chunk list → empty result |

### TestMergeEvaluationData (3 tests)

| Test | Scenario |
|---|---|
| `test_single_kb` | Single KB returns its questions |
| `test_two_kbs_merge_relevant_chunks` | Same question in 2 KBs → union of relevant chunks |
| `test_question_in_one_kb_only` | Different questions from different KBs both included |

### TestComputeAbstentionMetrics (4 tests)

| Test | Scenario |
|---|---|
| `test_perfect_classification` | Perfect: P=1.0, R=1.0, F1=1.0, FAR=0.0 |
| `test_all_false_positives` | All false positives: P=0.0 |
| `test_empty_input` | Empty → P=1.0, R=1.0, F1=1.0 |
| `test_mixed_results` | Mixed: P=0.5, R=0.5, F1=0.5, FAR=0.5, Coverage=0.5 |
