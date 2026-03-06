# Hybrid Retrieval Test Documentation

**Test file:** `tests/retrieval/test_hybrid.py`
**Total tests:** 23
**Run command:** `python -m pytest tests/retrieval/test_hybrid.py -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify RRF fusion logic, constructor validation, delegation to sub-retrievers, score properties, and edge cases using a `_FakeRetriever` helper. These tests run without external data files and are fast.
2. **Integration tests** run hybrid retrieval (BM25 + Vector) against the real chunk JSON files and pre-built vector stores, verifying that relevant results are returned. These tests are skipped automatically if the chunk files or vector stores are not present (`pytest.mark.skipif`).

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | What is verified |
|---|---|---|
| `TestHybridRetrieverUnit` | 15 | Constructor rejects ≠2 retrievers, `index()` delegates to both, `size` from first retriever, retrieve before index, result types, RRF fusion (shared docs rank higher), positive scores, descending order, k limit, custom k_rrf, re-indexing, metadata from highest-scoring source, no duplicates. |

### Integration Tests (require chunk JSON files + vector stores)

| Class | Tests | What is verified |
|---|---|---|
| `TestHybridStPOIntegration` | 3 | Corpus size >100, thesis query returns par-23 chunks, unique results. |
| `TestHybridFAQStPOIntegration` | 3 | Corpus size 999, thesis query returns results, FAQ metadata present. |
| `TestHybridFAQAOIntegration` | 2 | Corpus size 50, retrieve returns list. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `retriever` | class | Creates BM25 + Vector sub-retrievers, wraps them in HybridRetriever once per integration test class. |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not PATH.exists())`.
- Integration tests require **both** chunk JSON files **and** pre-built vector stores.
- If either is missing, only 15 unit tests run.
- Unit tests use `_FakeRetriever` and require no external dependencies.
- Chunk paths resolve to `{project_root}/data/chunks/*.json`.
- Vector store paths resolve to `{project_root}/data/vectorstore/{kb}/`.
