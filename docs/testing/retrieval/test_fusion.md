# RRF Fusion Utility Test Documentation

**Test file:** `tests/retrieval/test_fusion.py`
**Total tests:** 36 (36 unit)
**Run command:** `python -m pytest tests/retrieval/test_fusion.py -v`

---

## Test Strategy

All tests are pure unit tests using inline `RetrievalResult` objects. No external data files or retrieval indices are needed. Tests verify the mathematical correctness of the RRF fusion formula, weighted RRF, edge-case handling, metadata propagation, and output properties.

---

## Test Class: `TestRRFFuse`

### Edge Cases (3 tests)

| Test | What is verified |
|---|---|
| `test_empty_input_returns_empty` | Empty input list returns empty output. |
| `test_single_empty_list_returns_empty` | Single empty list returns empty output. |
| `test_multiple_empty_lists_returns_empty` | Multiple empty lists return empty output. |

### Single List (2 tests)

| Test | What is verified |
|---|---|
| `test_single_list_preserves_order` | Single input list preserves original rank order. |
| `test_single_list_respects_k` | Output is limited to k results. |

### Two Lists (3 tests)

| Test | What is verified |
|---|---|
| `test_two_lists_shared_document_ranks_higher` | A document appearing in both lists ranks higher than documents in only one. |
| `test_two_lists_no_overlap` | Non-overlapping lists produce union of all documents. |
| `test_two_lists_full_overlap` | Fully overlapping lists produce no duplicates. |

### Three or More Lists (2 tests)

| Test | What is verified |
|---|---|
| `test_three_lists_fuses_correctly` | Three-list fusion places the document appearing in all three at rank 1. |
| `test_three_lists_document_in_all_beats_document_in_two` | Document in 3 lists ranks above document in 2 lists. |

### RRF Score Computation (3 tests)

| Test | What is verified |
|---|---|
| `test_rrf_score_formula_two_lists` | Score matches `2 * 1/(k_rrf+1)` for document at rank 1 in both lists. |
| `test_rrf_score_formula_different_ranks` | Score matches `1/(k_rrf+1) + 1/(k_rrf+2)` for document at different ranks. |
| `test_custom_k_rrf_affects_scores` | Smaller k_rrf produces higher scores (k_rrf=1 vs k_rrf=60). |

### Metadata Handling (2 tests)

| Test | What is verified |
|---|---|
| `test_metadata_from_highest_scoring_source` | Text and metadata come from the list where the document had the highest original score. |
| `test_metadata_preserved_for_unique_documents` | Metadata is preserved for documents appearing in only one list. |

### Output Properties (5 tests)

| Test | What is verified |
|---|---|
| `test_scores_sorted_descending` | Output is sorted by descending RRF score. |
| `test_all_scores_positive` | All RRF scores are positive. |
| `test_no_duplicate_chunk_ids` | No duplicate chunk_ids in output (even with 3-list overlap). |
| `test_returns_retrieval_result_type` | All results are `RetrievalResult` instances. |
| `test_k_limits_output` | Output length does not exceed k. |

### Weighted RRF (6 tests)

| Test | What is verified |
|---|---|
| `test_uniform_weights_match_no_weights` | Explicit uniform weights produce same ranking as no weights. |
| `test_double_weight_boosts_retriever` | Weight 2.0 on second list boosts its unique document to rank 1. |
| `test_weights_wrong_length_raises` | Weights list with wrong length raises `ValueError`. |
| `test_weights_negative_raises` | Negative weight raises `ValueError`. |
| `test_weights_zero_raises` | Zero weight raises `ValueError`. |
| `test_weighted_score_formula` | Weighted score matches `weight / (k_rrf + rank)` formula exactly. |

---

## Test Class: `TestFusionRetriever` (10 tests)

Tests for the `FusionRetriever` wrapper class that fuses results across multiple pre-indexed retrievers.

| Test | What is verified |
|---|---|
| `test_single_sub_retriever_pass_through` | Single sub-retriever: results pass through unchanged. |
| `test_two_sub_retrievers_fuse_results` | Two sub-retrievers: results fused via RRF. |
| `test_k_limits_output` | Output limited to k results. |
| `test_size_sums_sub_retrievers` | `size` returns sum of all sub-retriever sizes. |
| `test_index_raises_not_implemented_error` | `index()` raises `NotImplementedError`. |
| `test_empty_retrievers_raises_value_error` | Empty retriever list raises `ValueError`. |
| `test_scores_are_rrf_scores` | Output scores are RRF-computed (not original scores). |
| `test_custom_k_rrf_applied` | Custom `k_rrf` parameter affects score computation. |
| `test_weights_passed_through` | Weights are forwarded to `rrf_fuse()` and affect ranking. |
| `test_weights_wrong_length_raises` | Weights list with wrong length raises `ValueError`. |

---

## CI Considerations

- All 36 tests are pure unit tests with no external dependencies.
- No `pytest.mark.skipif` guards needed.
- Fast execution (~0.01s total).
