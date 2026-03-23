# Server Test Documentation

**Test files:** `tests/server/test_models.py`, `tests/server/test_service.py`, `tests/server/test_api.py`, `tests/server/test_pipeline.py`
**Total tests:** 52 (52 unit)
**Run command:** `python -m pytest tests/server/ -v`

---

## Test Strategy

Server tests verify the Pydantic API models, PipelineService caching and orchestration, abstention pipeline, and all FastAPI endpoints using TestClient. All tests use stubbed retrievers and generators (`StubRetriever`, `StubGenerator`) to avoid external dependencies (no Ollama, no ChromaDB, no real indices).

---

## Test Files

| Test File | Tests | Description |
|---|---|---|
| `tests/server/test_models.py` | 15 | Pydantic model validation, config defaults |
| `tests/server/test_service.py` | 10 | PipelineService: caching, retrieval strategies, chat |
| `tests/server/test_api.py` | 17 | API endpoints: chat, health, options, UI pages |
| `tests/server/test_pipeline.py` | 10 | Abstention pipeline orchestration: Level 1 + Level 2 |
| **Total** | **52** | |

## Stub Design

All tests use two stubs that avoid real BM25/Vector/Ollama dependencies:

- **StubRetriever**: Returns BM25-style scores based on keyword overlap between query and document text. Implements the full `Retriever` interface.
- **StubGenerator**: Returns a fixed answer (`"The answer is 42."`) or an abstention response for configured keywords. Exposes `self.model = "stub-model"`.

The service tests monkey-patch `PipelineService._create_retriever` and `load_chunks` to inject stubs. The API tests additionally patch `OllamaGenerator` and `check_ollama`.

## Test Details

### tests/server/test_models.py (15 tests)

| Test Class | Tests | What is verified |
|---|---|---|
| `TestServerConfig` | 5 | Default values, KB paths match expected, normalization map covers all types, threshold map covers all normalizations |
| `TestChatRequest` | 5 | Defaults applied, query required, min_length, k range (1–50), threshold range (0–1) |
| `TestChatResponse` | 3 | Full response construction, abstention fields, sources list |
| `TestOptionsResponse` | 2 | Fields present, defaults dict included |

### tests/server/test_service.py (10 tests)

| Test Class | Tests | What is verified |
|---|---|---|
| `TestPipelineServiceCaching` | 3 | Same config returns same instance, different configs create different instances, cached count tracks correctly |
| `TestPipelineServiceRetriever` | 3 | Merged pool merges chunks, fusion creates FusionRetriever, unknown type raises ValueError |
| `TestPipelineServiceChat` | 3 | Complete response structure, abstention at high threshold (Level 1), normal answer at low threshold |
| `TestPipelineServiceMisc` | 1 | Generator model property |

### tests/server/test_api.py (17 tests)

| Test Class | Tests | What is verified |
|---|---|---|
| `TestChatPage` | 2 | GET / returns 200, response is HTML containing "MARley" |
| `TestDebugPage` | 2 | GET /debug returns 200, response is HTML containing "Debug" |
| `TestHealthEndpoint` | 2 | Returns status field, includes ollama field |
| `TestOptionsEndpoint` | 3 | Returns retriever_types, strategies, defaults dict |
| `TestChatEndpoint` | 8 | Normal answer, abstention at high threshold, empty query returns 400, sources included, confidence field, config in response, invalid retriever returns 422, whitespace trimmed |

### tests/server/test_pipeline.py (10 tests)

| Test Class | Tests | What is verified |
|---|---|---|
| `TestRunWithAbstention` | 10 | Level 1 abstention (low scores), Level 2 abstention (LLM abstains), normal answer, confidence computed correctly, filtered results passed to generator, threshold=0 no filtering, threshold=1 always abstains, BM25 normalization strategy, empty retrieval results, result fields complete |

---

## Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `StubRetriever` | — | Returns BM25-style scores based on keyword overlap. Implements full `Retriever` interface. |
| `StubGenerator` | — | Returns fixed answer or abstention response for configured keywords. |

---

## CI Considerations

- All 52 tests run without external dependencies.
- Service tests monkey-patch `PipelineService._create_retriever` and `load_chunks`.
- API tests additionally patch `OllamaGenerator` and `check_ollama`.
- No integration tests — Ollama integration is tested manually.
