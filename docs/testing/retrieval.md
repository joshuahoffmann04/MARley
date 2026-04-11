# Retrieval Tests

> Tests for all retrieval strategies in `src/marley/retrieval/`.

**Test files**: `test_bm25.py` (27), `test_vector.py` (25), `test_hybrid.py` (26), `test_fusion.py` (37), `test_merged.py` (14)
**Shared**: `tests/retrieval/conftest.py` (contract mixin + path constants)
**Total**: 129 tests

---

## Shared Infrastructure (conftest.py)

### RetrieverContractTests Mixin (12 tests)

Inherited by `TestBM25RetrieverUnit` and `TestVectorRetrieverUnit` to guarantee
that every `Retriever` implementation satisfies the interface contract.

| Test | Validates |
|------|-----------|
| `test_index_sets_size` | `index()` sets `size` to corpus length |
| `test_index_empty_corpus` | Empty corpus results in `size == 0` |
| `test_reindex_replaces_corpus` | Re-indexing replaces the previous corpus |
| `test_retrieve_before_index_returns_empty` | Retrieval before indexing returns `[]` |
| `test_retrieve_returns_results` | Retrieval returns non-empty results |
| `test_retrieve_returns_retrieval_result_type` | All results are `RetrievalResult` instances |
| `test_retrieve_ranked_by_score` | Results are sorted by score descending |
| `test_retrieve_top1_is_most_relevant` | Top-1 result is the most relevant document |
| `test_retrieve_respects_k` | Result count does not exceed `k` |
| `test_metadata_preserved` | Metadata survives indexing and retrieval |
| `test_retrieve_empty_query` | Empty query returns a list (no crash) |
| `test_retrieve_k_zero` | `k=0` returns empty list |

### Path Constants

- `STPO_CHUNKS_PATH`: `data/chunks/stpo-chunks.json`
- `FAQ_STPO_CHUNKS_PATH`: `data/chunks/faq-stpo-chunks.json`
- `FAQ_AO_CHUNKS_PATH`: `data/chunks/faq-ao-chunks.json`
- `VECTORSTORE_DIR`: `data/vectorstore/`

---

## test_bm25.py — BM25 Retriever

### TestTokenize (5 tests)

| Test | Validates |
|------|-----------|
| `test_lowercases` | Tokens are lowercased |
| `test_splits_whitespace` | Splits on whitespace (spaces, tabs, newlines) |
| `test_empty_string` | Empty string returns empty list |
| `test_preserves_punctuation` | Punctuation like `SS23?` is kept |
| `test_preserves_german_umlauts` | German umlauts (u, u, a) are preserved |

### TestBM25RetrieverUnit (14 tests = 12 contract + 2 BM25-specific)

Inherits `RetrieverContractTests`. Additional tests:

| Test | Validates |
|------|-----------|
| `test_retrieve_filters_zero_scores` | Zero-score documents are excluded |
| `test_index_validates_corpus` | `ValueError` for corpus missing required keys |

### Integration Tests (3 classes, 10 tests)

**TestBM25StPOIntegration** (3 tests, require stpo-chunks.json):
- `test_corpus_size` — 153 chunks indexed
- `test_thesis_query` — "master thesis" query finds par-23 chunks
- `test_unique_results` — No duplicate chunk IDs

**TestBM25FAQStPOIntegration** (3 tests, require faq-stpo-chunks.json):
- `test_corpus_size` — 1039 chunks indexed
- `test_thesis_query` — FAQ thesis query returns results
- `test_results_have_faq_metadata` — Results have `faq_source` metadata

**TestBM25FAQAOIntegration** (2 tests, require faq-ao-chunks.json):
- `test_corpus_size_placeholder` — AO placeholder has 0 chunks
- `test_retrieve_empty_corpus` — Empty corpus returns empty results

---

## test_vector.py — Vector Retriever

### TestVectorRetrieverUnit (17 tests = 12 contract + 5 Vector-specific)

Inherits `RetrieverContractTests`. Uses `tmp_path` fixture for ChromaDB persistence.

| Test | Validates |
|------|-----------|
| `test_score_range` | Cosine similarity scores are in [-1.0, 1.0] |
| `test_metadata_none_values_handled` | `None` metadata values are converted to `""` |
| `test_metadata_list_values_handled` | List metadata values are joined with ` > ` |
| `test_persistence_survives_new_instance` | New `VectorRetriever` instance loads persisted data |
| `test_size_zero_before_index` | `size` is 0 before indexing |

### Integration Tests (3 classes, 8 tests)

Same structure as BM25: StPO (3), FAQ-StPO (3), FAQ-AO (2) + class-scoped fixtures
using `tmp_path_factory` for isolated vector stores.

---

## test_hybrid.py — Hybrid BM25+Vector Retriever

### TestHybridRetrieverUnit (18 tests)

Uses `_FakeRetriever` stubs for deterministic testing.

| Test | Validates |
|------|-----------|
| `test_requires_two_retrievers` | `ValueError` for 1 retriever |
| `test_requires_two_retrievers_three` | `ValueError` for 3 retrievers |
| `test_index_delegates_to_both` | `index()` calls both sub-retrievers |
| `test_size_returns_first_retriever_size` | `size` from first sub-retriever |
| `test_retrieve_before_index_returns_empty` | Empty results before indexing |
| `test_retrieve_returns_retrieval_result_type` | Returns `RetrievalResult` instances |
| `test_retrieve_fuses_results` | Documents from both retrievers are merged |
| `test_rrf_scores_are_positive` | All RRF scores are positive |
| `test_rrf_scores_sorted_descending` | Results sorted by RRF score |
| `test_respects_k` | Output limited to `k` results |
| `test_document_in_both_ranks_higher` | Document appearing in both retrievers ranks first |
| `test_custom_k_rrf` | Custom `k_rrf` changes score magnitude |
| `test_reindex_replaces_corpus` | Re-indexing replaces corpus in both sub-retrievers |
| `test_metadata_from_highest_scoring_source` | Text/metadata from highest-scoring source |
| `test_no_duplicate_results` | No duplicate chunk IDs in output |
| `test_weights_boost_second_retriever` | Weight [1.0, 2.0] boosts second retriever |
| `test_uniform_weights_match_default` | Explicit [1.0, 1.0] matches no-weights default |
| `test_weights_wrong_length_raises` | `ValueError` for wrong weight count |

### Integration Tests (3 classes, 8 tests)

StPO (3), FAQ-StPO (3), FAQ-AO (2). Require both chunk files and pre-built
vectorstores under `data/vectorstore/`.

---

## test_fusion.py — RRF Fusion and FusionRetriever

### TestRRFFuse (26 tests)

Tests for the standalone `rrf_fuse()` function.

**Edge cases** (4 tests):
- `test_empty_input_returns_empty`, `test_single_empty_list_returns_empty`
- `test_multiple_empty_lists_returns_empty`, `test_mixed_empty_and_nonempty_lists`

**Single list** (2 tests):
- `test_single_list_preserves_order`, `test_single_list_respects_k`

**Two lists** (3 tests):
- `test_two_lists_shared_document_ranks_higher`
- `test_two_lists_no_overlap`, `test_two_lists_full_overlap`

**Three+ lists** (2 tests):
- `test_three_lists_fuses_correctly`
- `test_three_lists_document_in_all_beats_document_in_two`

**RRF score computation** (3 tests):
- `test_rrf_score_formula_two_lists` — verifies `2/61` for k_rrf=60
- `test_rrf_score_formula_different_ranks` — verifies rank-based scoring
- `test_custom_k_rrf_affects_scores`

**Metadata handling** (2 tests):
- `test_metadata_from_highest_scoring_source`
- `test_metadata_preserved_for_unique_documents`

**Output properties** (5 tests):
- `test_scores_sorted_descending`, `test_all_scores_positive`
- `test_no_duplicate_chunk_ids`, `test_returns_retrieval_result_type`
- `test_k_limits_output`

**Weighted RRF** (6 tests):
- `test_uniform_weights_match_no_weights`, `test_double_weight_boosts_retriever`
- `test_weights_wrong_length_raises`, `test_weights_negative_raises`, `test_weights_zero_raises`
- `test_weighted_score_formula` — verifies `4.0/61` for weight [1.0, 3.0]

### TestFusionRetriever (10 tests)

Tests for the `FusionRetriever` wrapper class using `_StubRetriever`.

| Test | Validates |
|------|-----------|
| `test_single_sub_retriever_pass_through` | Single sub-retriever results pass through |
| `test_two_sub_retrievers_fuse_results` | Two sub-retrievers are fused via RRF |
| `test_k_limits_output` | Output limited to `k` |
| `test_size_sums_sub_retrievers` | `size` is sum of sub-retriever sizes |
| `test_index_raises_not_implemented_error` | `index()` raises (sub-retrievers are pre-indexed) |
| `test_empty_retrievers_raises_value_error` | Empty retriever list raises `ValueError` |
| `test_scores_are_rrf_scores` | Scores match RRF formula |
| `test_custom_k_rrf_applied` | Custom `k_rrf` is applied |
| `test_weights_passed_through` | Weights are forwarded to `rrf_fuse` |
| `test_weights_wrong_length_raises` | `ValueError` for mismatched weight count |

---

## test_merged.py — Merged-Pool Retriever

### TestMergedRetrieverUnit (10 tests)

Uses `_FakeRetriever` for deterministic testing.

| Test | Validates |
|------|-----------|
| `test_index_delegates_to_inner` | `index()` delegates to inner retriever |
| `test_size_delegates_to_inner` | `size` delegates to inner retriever |
| `test_retrieve_delegates_to_inner` | `retrieve()` delegates to inner retriever |
| `test_retrieve_respects_k` | Output limited to `k` |
| `test_retrieve_before_index_returns_empty` | Empty results before indexing |
| `test_index_empty_corpus` | Empty corpus results in `size == 0` |
| `test_reindex_replaces_corpus` | Re-indexing replaces corpus |
| `test_returns_retrieval_result_type` | Returns `RetrievalResult` instances |
| `test_merged_corpus_from_multiple_kbs` | Simulates merging chunks from two KBs |
| `test_inner_retriever_is_accessible` | Inner retriever state is shared |

### TestMergedBM25Integration (4 tests, require stpo + faq-stpo chunks)

| Test | Validates |
|------|-----------|
| `test_corpus_size` | 153 + 1039 = 1192 merged chunks |
| `test_thesis_query_returns_results` | Query returns results from merged pool |
| `test_results_from_both_kbs` | Results include chunks from at least one KB |
| `test_unique_results` | No duplicate chunk IDs |
