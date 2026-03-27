# MergedRetriever Test Documentation

**Test file:** `tests/retrieval/test_merged.py`
**Total tests:** 14 (10 unit, 4 integration)
**Run command:** `python -m pytest tests/retrieval/test_merged.py -v`

---

## Test Strategy

Unit tests use a `_FakeRetriever` stub to verify that MergedRetriever correctly delegates all operations to its inner retriever. Integration tests use real BM25 indexing over merged StPO + FAQ-StPO chunks.

---

## Test Class: `TestMergedRetrieverUnit` (10 tests)

| Test | What is verified |
|---|---|
| `test_index_delegates_to_inner` | Corpus passed to `index()` reaches the inner retriever unchanged. |
| `test_size_delegates_to_inner` | `size` reflects the inner retriever's corpus size. |
| `test_retrieve_delegates_to_inner` | `retrieve()` returns the inner retriever's results. |
| `test_retrieve_respects_k` | Output is limited to k results. |
| `test_retrieve_before_index_returns_empty` | Retrieve on unindexed retriever returns empty list. |
| `test_index_empty_corpus` | Empty corpus results in size 0. |
| `test_reindex_replaces_corpus` | Re-indexing replaces the previous corpus. |
| `test_returns_retrieval_result_type` | All results are `RetrievalResult` instances. |
| `test_merged_corpus_from_multiple_kbs` | Simulates merging two KBs before indexing; verifies both KBs present. |
| `test_inner_retriever_is_accessible` | Inner retriever state reflects operations on the wrapper. |

---

## Test Class: `TestMergedBM25Integration` (4 tests)

Requires `data/chunks/stpo-chunks.json` and `data/chunks/faq-stpo-chunks.json`.

| Test | What is verified |
|---|---|
| `test_corpus_size` | Merged corpus size equals StPO (153) + FAQ-StPO (1039) = 1192. |
| `test_thesis_query_returns_results` | Thesis query returns non-empty results. |
| `test_results_from_both_kbs` | Results include chunks from at least one of the two KBs. |
| `test_unique_results` | No duplicate chunk IDs in results. |

---

## CI Considerations

- Unit tests (10) require no external data files.
- Integration tests (4) are marked with `@pytest.mark.integration` and `@pytest.mark.skipif` for missing chunk files.
