# Testing Overview

> Test strategy, conventions, and execution for the MARley test suite.

## Test Suite Summary

| Module | Test File(s) | Tests |
|--------|-------------|-------|
| Models | test_models.py, test_scoring.py | 54 |
| Extractor | test_extractor.py | 81 |
| Chunker | test_pdf_chunker.py, test_faq_chunker.py | 95 |
| Retrieval | test_bm25.py, test_vector.py, test_hybrid.py, test_fusion.py, test_merged.py | 129 |
| Generator | test_generator.py | 24 |
| Abstention | test_detection.py | 12 |
| Server | test_api.py, test_models.py, test_pipeline.py, test_service.py | 52 |
| **Total** | **16 files** | **447** |

## Test Strategy

### Unit Tests First

The test suite follows a **stubs-over-mocks** principle. Instead of patching
internals with `unittest.mock`, test modules use lightweight stub classes that
implement the same abstract base classes as the production code.  This gives
realistic behavior without external dependencies (Ollama, ChromaDB, data files).

Core stubs are defined once in `tests/conftest.py`:

| Stub | Base Class | Behavior |
|------|-----------|----------|
| `KeywordRetriever` | `Retriever` | Scores documents by query-word overlap. Mimics BM25-style ranking. |
| `FixedRetriever` | `Retriever` | Returns pre-configured results. Full control over retrieval output. |
| `StubGenerator` | `Generator` | Returns a fixed answer. Optionally abstains on configured keywords. |

### Integration Tests

Tests that require real data files or external services (Ollama, ChromaDB with
real embeddings) are marked with `@pytest.mark.integration`.  They are **not
skipped by default** but use `pytest.mark.skipif` guards that skip gracefully
when prerequisites are unavailable.

Integration tests exist for: extractor (PDF), chunker (StPO/FAQ data files),
retrieval (real chunk files + vector stores), and generator (Ollama server).

### Contract Tests

The retrieval module uses a **contract test mixin** (`RetrieverContractTests`
in `tests/retrieval/conftest.py`) that defines 12 interface tests every
`Retriever` implementation must satisfy.  BM25 and Vector test modules inherit
this mixin, guaranteeing behavioral consistency across strategies.

## Test Conventions

### File Structure

```
tests/
  conftest.py               # Shared stubs (KeywordRetriever, FixedRetriever, StubGenerator)
  models/
    test_models.py           # Dataclass construction, validation, edge cases
    test_scoring.py          # Scoring functions (RRF, confidence, normalization)
  extractor/
    test_extractor.py        # PDF text + table extraction logic
  chunker/
    test_pdf_chunker.py      # Sentence-level chunking with overlap
    test_faq_chunker.py      # FAQ chunking (Q&A pair extraction)
  retrieval/
    conftest.py              # RetrieverContractTests mixin, shared paths
    test_bm25.py             # BM25 retriever (unit + integration)
    test_vector.py           # Vector retriever (unit + integration)
    test_hybrid.py           # Hybrid BM25+Vector via RRF
    test_fusion.py           # Cross-knowledge-base fusion
    test_merged.py           # Merged-pool strategy
  generator/
    test_generator.py        # Ollama generator (unit + integration)
  abstention/
    test_detection.py        # Two-level abstention detection
  server/
    test_api.py              # FastAPI endpoint tests (TestClient)
    test_models.py           # Pydantic request/response models
    test_pipeline.py         # Pipeline orchestration logic
    test_service.py          # PipelineService (caching, chat)
```

### Naming Conventions

- **Test files**: `test_<module>.py` — mirrors the source module name
- **Test classes**: `Test<Feature>` — groups related test cases (e.g., `TestRetrievalResult`)
- **Test functions**: `test_<behavior>` — describes the expected behavior
- **Parametrized tests**: Use `@pytest.mark.parametrize` for input variations

### Shared Test Data

A canonical `SMALL_CORPUS` is defined in `tests/conftest.py` with 3 documents
covering typical StPO topics (thesis credits, examination rules, study abroad).
This corpus is reused across retrieval tests via import.

## Markers

| Marker | Purpose |
|--------|---------|
| `integration` | Requires data files or external services (Ollama, ChromaDB) |

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Running Tests

```bash
# Full test suite (unit + integration where available)
python -m pytest tests/ -v

# Unit tests only (skip integration)
python -m pytest tests/ -m "not integration"

# Specific module
python -m pytest tests/retrieval/ -v

# Specific file
python -m pytest tests/retrieval/test_bm25.py -v

# With coverage
python -m pytest tests/ --cov=src/marley --cov-report=term-missing

# Quick run (default: short traceback, quiet)
python -m pytest tests/
```

### pytest Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "evaluation/tests"]
python_files = "test_*.py"
addopts = "--strict-markers --tb=short -q"
markers = [
    "integration: requires data files or external services (Ollama, ChromaDB)",
]
```

## Fixture Architecture

### Root conftest.py (tests/)

Provides the three canonical stubs used throughout the suite:

- **`KeywordRetriever`**: Scores by query-word overlap x configurable multiplier.
  Tests import and instantiate it directly (not as a fixture).
- **`FixedRetriever`**: Returns pre-configured `RetrievalResult` list.
  Used when tests need exact control over retrieval output.
- **`StubGenerator`**: Returns a fixed answer string. Optionally abstains
  when the query contains configured keywords. Used in pipeline and server tests.
- **`SMALL_CORPUS`**: 3-document list for retrieval indexing.

### Retrieval conftest.py (tests/retrieval/)

- **`RetrieverContractTests`**: Mixin class with 12 contract tests for the
  `Retriever` interface (index, retrieve, ranking, metadata, edge cases).
- **Path constants**: `STPO_CHUNKS_PATH`, `FAQ_STPO_CHUNKS_PATH`,
  `FAQ_AO_CHUNKS_PATH`, `VECTORSTORE_DIR` for integration tests.

## Per-Module Documentation

Detailed test documentation for each module:

| Document | Covers |
|----------|--------|
| [models.md](models.md) | Dataclasses, scoring functions, constants |
| [extractor.md](extractor.md) | PDF text + table extraction |
| [chunker.md](chunker.md) | StPO + FAQ chunking |
| [retrieval.md](retrieval.md) | BM25, Vector, Hybrid, Fusion, Merged |
| [generator.md](generator.md) | Ollama LLM generation |
| [abstention.md](abstention.md) | Two-level abstention detection |
| [server.md](server.md) | FastAPI endpoints, pipeline, service |
