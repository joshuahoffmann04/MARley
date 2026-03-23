# Vector Retrieval Test Documentation

**Test file:** `tests/retrieval/test_vector.py`
**Total tests:** 25 (17 unit + 8 integration)
**Run command:** `python -m pytest tests/retrieval/test_vector.py -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify the VectorRetriever using synthetic corpora and temporary directories. The `TestVectorRetrieverUnit` class inherits from `RetrieverContractTests` (defined in `tests/retrieval/conftest.py`), which provides 12 shared interface contract tests. Vector adds 5 specific tests for score range, metadata flattening, persistence, and initial size.
2. **Integration tests** run vector retrieval against the real chunk JSON files and verify that relevant results are returned. These tests are skipped automatically if the chunk files are not present (`pytest.mark.skipif`).

All tests use `tmp_path` / `tmp_path_factory` fixtures for ChromaDB storage, ensuring no test state leaks between runs.

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | What is verified |
|---|---|---|
| `TestVectorRetrieverUnit` | 17 | 12 contract tests + score range [-1, 1], None metadata flattening, list metadata joining, persistence across instances, size zero before index. |

### Integration Tests (require chunk JSON files)

| Class | Tests | What is verified |
|---|---|---|
| `TestVectorStPOIntegration` | 3 | Corpus size == 153, thesis query returns par-23 chunks, unique results. |
| `TestVectorFAQStPOIntegration` | 3 | Corpus size == 1039, thesis query returns results, FAQ metadata present. |
| `TestVectorFAQAOIntegration` | 2 | Placeholder produces 0 chunks, empty corpus returns empty list. |

---

## Contract Test Mixin

The `RetrieverContractTests` class in `tests/retrieval/conftest.py` defines 12 tests that verify the `Retriever` interface contract. Both `TestBM25RetrieverUnit` and `TestVectorRetrieverUnit` inherit from it, eliminating test duplication while ensuring consistent interface verification.

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `_setup` | function (autouse) | Creates a temporary persist directory for each unit test. |
| `retriever` | class | Loads chunks, builds vector index once per integration test class (using `tmp_path_factory`). |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not PATH.exists())`.
- If chunk JSON files are not available, only 17 unit tests run.
- Unit tests use temporary directories and do not require GPU — CPU inference is sufficient.
- Chunk paths resolve to `{project_root}/data/chunks/*.json`.
