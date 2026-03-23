# Hybrid Retrieval Test Documentation

**Test file:** `tests/retrieval/test_hybrid.py`
**Total tests:** 26 (18 unit + 8 integration)
**Run command:** `python -m pytest tests/retrieval/test_hybrid.py -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify RRF fusion logic, constructor validation, delegation to sub-retrievers, score properties, weighted fusion, and edge cases using a `_FakeRetriever` helper. These tests run without external data files and are fast.
2. **Integration tests** run hybrid retrieval (BM25 + Vector) against the real chunk JSON files and pre-built vector stores, verifying that relevant results are returned. These tests are skipped automatically if the chunk files or vector stores are not present (`pytest.mark.skipif`).

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | What is verified |
|---|---|---|
| `TestHybridRetrieverUnit` | 18 | Constructor rejects !=2 retrievers (2 tests), `index()` delegates to both, `size` from first retriever, retrieve before index, result types, RRF fusion (shared docs rank higher), positive scores, descending order, k limit, document in both ranks higher, custom k_rrf, re-indexing, metadata from highest-scoring source, no duplicates, weights boost second retriever, uniform weights match default, weights wrong length raises. |

### Integration Tests (require chunk JSON files + vector stores)

| Class | Tests | What is verified |
|---|---|---|
| `TestHybridStPOIntegration` | 3 | Corpus size == 153, thesis query returns par-23 chunks, unique results. |
| `TestHybridFAQStPOIntegration` | 3 | Corpus size == 1039, thesis query returns results, FAQ metadata present. |
| `TestHybridFAQAOIntegration` | 2 | Placeholder produces 0 chunks, empty corpus returns empty list. Skipped until FAQ-AO vectorstore exists. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `retriever` | class | Creates BM25 + Vector sub-retrievers, wraps them in HybridRetriever once per integration test class. |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not PATH.exists())`.
- Integration tests require **both** chunk JSON files **and** pre-built vector stores.
- If either is missing, only 18 unit tests run.
- FAQ-AO integration tests (2) are permanently skipped until FAQ-AO content is available and vectorized.
- Unit tests use `_FakeRetriever` and require no external dependencies.
- Chunk paths resolve to `{project_root}/data/chunks/*.json`.
- Vector store paths resolve to `{project_root}/data/vectorstore/{kb}/`.
