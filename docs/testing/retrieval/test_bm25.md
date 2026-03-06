# BM25 Retrieval Test Documentation

**Test file:** `tests/retrieval/test_bm25.py`
**Total tests:** 23
**Run command:** `python -m pytest tests/retrieval/test_bm25.py -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify tokenization and the BM25 retriever using synthetic corpora. The `TestBM25RetrieverUnit` class inherits from `RetrieverContractTests` (defined in `tests/retrieval/conftest.py`), which provides 10 shared interface contract tests. BM25 adds one specific test for zero-score filtering.
2. **Integration tests** run BM25 retrieval against the real chunk JSON files and verify that relevant results are returned. These tests are skipped automatically if the chunk files are not present (`pytest.mark.skipif`).

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | What is verified |
|---|---|---|
| `TestTokenize` | 4 | Lowercasing, whitespace splitting, empty string, punctuation preservation. |
| `TestBM25RetrieverUnit` | 11 | 10 contract tests (index size, empty corpus, re-indexing, retrieve before index, result types, ranking, top-1 relevance, k limit, metadata) + BM25-specific zero-score filtering. |

### Integration Tests (require chunk JSON files)

| Class | Tests | What is verified |
|---|---|---|
| `TestBM25StPOIntegration` | 3 | Corpus size >100, thesis query returns §23 chunks, unique results. |
| `TestBM25FAQStPOIntegration` | 3 | Corpus size 999, thesis query returns results, FAQ metadata present. |
| `TestBM25FAQAOIntegration` | 2 | Corpus size 50, retrieve returns list. |

---

## Contract Test Mixin

The `RetrieverContractTests` class in `tests/retrieval/conftest.py` defines 10 tests that verify the `Retriever` interface contract. Both `TestBM25RetrieverUnit` and `TestVectorRetrieverUnit` inherit from it, eliminating test duplication while ensuring consistent interface verification.

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `retriever` | class | Loads chunks and builds BM25 index once per integration test class. |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not PATH.exists())`.
- If chunk JSON files are not available, only 15 unit tests run.
- Chunk paths resolve to `{project_root}/data/chunks/*.json`.
