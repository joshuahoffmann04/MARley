# Combined Retrieval Evaluation Test Documentation

**Test file:** `evaluation/tests/retrieval/test_combined.py`
**Total tests:** 25
**Run command:** `python -m pytest evaluation/tests/retrieval/test_combined.py -v`

---

## Test Strategy

All tests are pure unit tests using synthetic data. No external files, knowledge bases, or vector stores are required. Tests verify the correctness of chunk merging, evaluation data merging, and both evaluation runner strategies using a `_StubRetriever` that returns the first k indexed chunks for any query.

---

## Test Classes

### Merge Chunks Tests (`TestMergeChunks`, 6 tests)

| Test | What is verified |
|---|---|
| `test_merges_two_files` | Two chunk files are correctly concatenated, all chunk_ids present. |
| `test_merges_three_files` | Three chunk files produce correct total count. |
| `test_preserves_order` | Chunks maintain insertion order (KB_1 chunks before KB_2). |
| `test_detects_duplicate_chunk_ids` | `ValueError` raised when same chunk_id appears in multiple files. |
| `test_empty_files` | Empty chunk file returns empty list. |
| `test_single_file` | Single file input works correctly. |

### Merge Evaluation Data Tests (`TestMergeEvaluationData`, 6 tests)

| Test | What is verified |
|---|---|
| `test_merges_relevant_chunks_across_kbs` | Relevant chunks from two KBs are merged via set union. |
| `test_question_in_only_one_kb` | Question with relevant chunks in only one KB retains those chunks. |
| `test_unanswerable_question_preserved` | `expected_abstention: true` is preserved through merging. |
| `test_deduplicates_relevant_chunks` | Overlapping chunk_ids are deduplicated. |
| `test_multiple_questions` | Multiple questions are all present in output. |
| `test_three_kbs` | Three KB evaluation files merge correctly. |

### Merged Pool Evaluation Tests (`TestRunMergedPoolEvaluation`, 6 tests)

| Test | What is verified |
|---|---|
| `test_returns_report_dict` | Report contains `strategy`, `metrics`, and `config` keys. |
| `test_correct_corpus_size` | Merged corpus size matches sum of all KB chunk counts. |
| `test_combination_field` | `combination` field is sorted, `+`-joined KB names. |
| `test_skips_unanswerable` | Unanswerable questions produce `num_queries=0`. |
| `test_metrics_are_valid` | All metric values are in [0.0, 1.0]. |
| `test_zero_chunk_kb_included` | A KB with 0 chunks (e.g., FAQ-AO) contributes nothing but does not break evaluation. |

### Fusion Evaluation Tests (`TestRunFusionEvaluation`, 7 tests)

| Test | What is verified |
|---|---|
| `test_returns_report_dict` | Report contains `strategy`, `metrics`, `config`, and `k_rrf`. |
| `test_correct_corpus_size` | Total corpus size matches sum of all KB chunk counts. |
| `test_combination_field` | `combination` field is sorted, `+`-joined KB names. |
| `test_skips_unanswerable` | Unanswerable questions produce `num_queries=0`. |
| `test_metrics_are_valid` | All metric values are in [0.0, 1.0]. |
| `test_retriever_type_in_config` | `config.retriever_type` reflects the factory's class name. |
| `test_zero_chunk_kb_included` | A KB with 0 chunks (e.g., FAQ-AO) contributes nothing but does not break fusion. |

---

## Fixtures

| Fixture / Helper | Description |
|---|---|
| `_write_chunks(path, chunks)` | Write a minimal chunk JSON file to disk. |
| `_write_eval(path, questions, kb)` | Write a minimal evaluation JSON file to disk. |
| `_chunk(cid, text)` | Create a minimal chunk dict. |
| `_StubRetriever` | Retriever that returns the first k indexed chunks for any query (score = 1/(rank+1)). |
| `tmp_path` | Pytest built-in, used for file I/O tests. |

---

## CI Considerations

- All tests run without external data files.
- No `pytest.mark.skipif` guards needed.
- Tests use `tmp_path` fixtures for file I/O tests.
