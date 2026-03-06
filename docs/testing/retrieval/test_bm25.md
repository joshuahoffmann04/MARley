# BM25 Retrieval Test Documentation

**Test file:** `tests/retrieval/test_bm25.py`
**Total tests:** 23
**Run command:** `python -m pytest tests/retrieval/ -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify tokenization, index building, retrieval behavior, and edge cases using synthetic corpora. These tests run without external data files and are fast.
2. **Integration tests** run BM25 retrieval against the real chunk JSON files and verify that relevant results are returned. These tests are skipped automatically if the chunk files are not present (`pytest.mark.skipif`).

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | Functions covered |
|---|---|---|
| `TestTokenize` | 4 | `_tokenize` |
| `TestBM25RetrieverUnit` | 11 | `BM25Retriever.index`, `BM25Retriever.retrieve`, `BM25Retriever.size` |

### Integration Tests (require chunk JSON files)

| Class | Tests | What is verified |
|---|---|---|
| `TestBM25StPOIntegration` | 3 | Corpus size >100, thesis query returns §23 chunks, unique results. |
| `TestBM25FAQStPOIntegration` | 3 | Corpus size 999, thesis query returns results, FAQ metadata present. |
| `TestBM25FAQAOIntegration` | 2 | Corpus size 50, retrieve returns list. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `retriever` | class | Loads chunks and builds BM25 index once per test class. |
| `chunks` | class | Raw chunk list from JSON. |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not PATH.exists())`.
- If chunk JSON files are not available, only 15 unit tests run.
- Chunk paths resolve to `{project_root}/data/chunks/*.json`.
