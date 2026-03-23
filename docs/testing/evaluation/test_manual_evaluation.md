# Manual Evaluation Test Documentation

**Test files:** `evaluation/tests/manual/test_models.py`, `evaluation/tests/manual/test_prepare.py`, `evaluation/tests/manual/test_metrics.py`
**Total tests:** 46 (46 unit)
**Run command:** `python -m pytest evaluation/tests/manual/ -v`

---

## Test Strategy

Tests cover the manual evaluation data model, item preparation, and metric computation. The FastAPI app is not unit-tested — it is a thin layer over the fully tested data model and metrics modules.

---

## test_models.py (23 tests)

Tests for the Judgement enum, EvaluationItem, ManualJudgement, and I/O functions.

### TestJudgement (9 tests)

| Test | Description |
|---|---|
| `test_all_six_values_exist` | Enum has exactly 6 members |
| `test_string_values` | Each enum value matches its expected string |
| `test_is_str_enum` | Judgement is a str subclass (JSON-serializable) |
| `test_answer_judgements_group` | ANSWER_JUDGEMENTS contains correct, partially_correct, incorrect |
| `test_abstention_judgements_group` | ABSTENTION_JUDGEMENTS contains the 3 abstention values |
| `test_groups_are_disjoint` | Answer and abstention groups share no members |
| `test_groups_cover_all` | Both groups together cover all 6 values |
| `test_construct_from_string` | `Judgement("correct")` returns `Judgement.CORRECT` |
| `test_invalid_value_raises` | Invalid string raises `ValueError` |

### TestEvaluationItem (2 tests)

| Test | Description |
|---|---|
| `test_create_with_all_fields` | Item creation with all fields populated |
| `test_default_metadata` | Metadata defaults to empty dict |

### TestManualJudgement (4 tests)

| Test | Description |
|---|---|
| `test_create_with_enum` | Creation with Judgement enum value |
| `test_auto_timestamp` | Timestamp auto-generated when not provided |
| `test_string_coercion` | String value auto-converted to Judgement enum |
| `test_preserves_explicit_timestamp` | Explicit timestamp not overwritten |

### TestSaveLoadItems (3 tests)

| Test | Description |
|---|---|
| `test_round_trip` | Save then load preserves all item fields |
| `test_file_contains_metadata` | Saved file includes metadata dict |
| `test_creates_parent_directory` | Missing parent directories created automatically |

### TestSaveLoadJudgements (5 tests)

| Test | Description |
|---|---|
| `test_save_creates_file` | First save creates the judgements file |
| `test_save_appends` | Multiple saves append to the same file |
| `test_load_deduplicates_by_latest` | Duplicate item_id: latest judgement wins |
| `test_load_nonexistent_returns_empty` | Missing file returns empty list |
| `test_metadata_updated` | File metadata tracks started and last_updated times |

---

## test_prepare.py (10 tests)

Tests for converting generation results into evaluation items.

### TestPrepareGenerationItems (7 tests)

| Test | Description |
|---|---|
| `test_correct_item_count` | 3 results produce 3 items |
| `test_item_id_format` | IDs follow `gen-{kb}-{qid}-d{n}` format |
| `test_metadata_fields` | All metadata fields populated correctly |
| `test_generated_and_reference_answer` | Answers preserved from source data |
| `test_enriches_from_eval_dataset` | Question text, category, expected_abstention from dataset |
| `test_without_eval_dataset_defaults` | Defaults when no dataset path provided |
| `test_all_items_are_evaluation_items` | All returned objects are EvaluationItem instances |

### TestPrepareItemsFromResults (3 tests)

| Test | Description |
|---|---|
| `test_correct_count` | Correct number of items from raw results |
| `test_with_question_metadata` | Question metadata enriches items correctly |
| `test_without_metadata_defaults` | Defaults when no metadata provided |

---

## test_metrics.py (13 tests)

Tests for `compute_manual_metrics()` covering all metric variants.

### TestComputeManualMetrics (13 tests)

| Test | Description |
|---|---|
| `test_empty_judgements` | Zero judgements: accuracy 0.0, counts correct |
| `test_all_correct` | All correct: strict=1.0, lenient=1.0 |
| `test_mixed_judgements_strict_vs_lenient` | 2 correct + 1 partial + 1 incorrect: strict=0.5, lenient=0.75 |
| `test_all_incorrect` | All incorrect: both accuracies 0.0 |
| `test_accuracy_by_distractor_level` | Per-level accuracy computed correctly |
| `test_distractor_levels_sorted` | Distractor levels in ascending order |
| `test_abstention_precision` | 1 correct + 1 incorrect abstention: precision=0.5 |
| `test_abstention_recall` | 1 correct + 1 missing abstention: recall=0.5 |
| `test_judgement_distribution` | Per-value count correct |
| `test_only_judged_items_count` | Pending items excluded from accuracy |
| `test_latest_judgement_wins` | Deduplicated judgement used for metrics |
| `test_knowledge_base_preserved` | KB label passed through to metrics |
| `test_abstention_metrics_zero_when_no_abstentions` | No abstention judgements: precision/recall both 0.0 |

---

## Test Coverage by Feature

| Feature | Tests | Test file |
|---|---|---|
| Judgement enum | 9 | `test_models.py` |
| Data classes (Item, Judgement) | 6 | `test_models.py` |
| I/O (save/load items + judgements) | 8 | `test_models.py` |
| Item preparation (from file) | 7 | `test_prepare.py` |
| Item preparation (from results) | 3 | `test_prepare.py` |
| Strict/lenient accuracy | 4 | `test_metrics.py` |
| Accuracy by distractor level | 2 | `test_metrics.py` |
| Abstention precision/recall | 3 | `test_metrics.py` |
| Judgement distribution | 1 | `test_metrics.py` |
| Edge cases (empty, defaults) | 3 | `test_metrics.py` |
| **Total** | **46** | |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `tmp_path` | function | Pytest built-in, used for save/load round-trip tests. |

---

## CI Considerations

- All 46 tests are pure unit tests with no external dependencies.
- No Ollama, ChromaDB, or real evaluation data required.
- Fast execution (~0.02s total).
