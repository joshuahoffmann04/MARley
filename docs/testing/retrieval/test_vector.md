# Vector Retrieval Test Documentation

**Test file:** `tests/retrieval/test_vector.py`
**Total tests:** 23
**Run command:** `python -m pytest tests/retrieval/test_vector.py -v`

---

## Test Strategy

Tests are organized into two categories:

1. **Unit tests** verify index building, retrieval behavior, score ranges, metadata handling, and persistence using synthetic corpora and temporary directories. These tests run without external data files.
2. **Integration tests** run vector retrieval against the real chunk JSON files and verify that relevant results are returned. These tests are skipped automatically if the chunk files are not present (`pytest.mark.skipif`).

All tests use `tmp_path` / `tmp_path_factory` fixtures for ChromaDB storage, ensuring no test state leaks between runs.

---

## Test Classes

### Unit Tests (no external data required)

| Class | Tests | What is verified |
|---|---|---|
| `TestVectorRetrieverUnit` | 15 | Index size, empty corpus, retrieve before index, result types, ranking order, top-1 relevance, k limit, score range [-1, 1], re-indexing, metadata preservation, None/list metadata flattening, persistence across instances, size property. |

### Integration Tests (require chunk JSON files)

| Class | Tests | What is verified |
|---|---|---|
| `TestVectorStPOIntegration` | 3 | Corpus size >100, thesis query returns par-23 chunks, unique results. |
| `TestVectorFAQStPOIntegration` | 3 | Corpus size 999, thesis query returns results, FAQ metadata present. |
| `TestVectorFAQAOIntegration` | 2 | Corpus size 50, retrieve returns list. |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `_setup_teardown` | function (autouse) | Creates a temporary persist directory for each unit test. |
| `retriever` | class | Loads chunks, builds vector index once per integration test class (using `tmp_path_factory`). |

---

## CI Considerations

- All integration tests are guarded by `pytest.mark.skipif(not PATH.exists())`.
- If chunk JSON files are not available, only 15 unit tests run.
- Unit tests use temporary directories and do not require GPU — CPU inference is sufficient.
- Chunk paths resolve to `{project_root}/data/chunks/*.json`.
