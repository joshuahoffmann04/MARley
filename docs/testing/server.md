# Server Tests

> Tests for the FastAPI server, Pydantic models, pipeline orchestration, and service layer in `src/marley/server/`.

**Test files**: `test_api.py` (17), `test_models.py` (15), `test_pipeline.py` (10), `test_service.py` (10)
**Total**: 52 tests

---

## test_api.py — FastAPI Endpoint Tests

Uses `fastapi.testclient.TestClient` with patched retrievers and generator.

**Fixture**: `client` — creates a `TestClient` with `KeywordRetriever` and `StubGenerator` stubs
via `@patch`. Also patches `load_chunks` and `check_ollama`.

### TestChatPage (2 tests)

| Test | Validates |
|------|-----------|
| `test_returns_200` | `GET /` returns HTTP 200 |
| `test_returns_html` | Response is HTML containing "MARley" |

### TestDebugPage (2 tests)

| Test | Validates |
|------|-----------|
| `test_returns_200` | `GET /debug` returns HTTP 200 |
| `test_returns_html` | Response is HTML containing "Debug" |

### TestHealthEndpoint (2 tests)

| Test | Validates |
|------|-----------|
| `test_returns_status` | `GET /api/health` returns JSON with `status` field |
| `test_includes_ollama_field` | Response includes `ollama` field ("connected" or "unavailable") |

### TestOptionsEndpoint (3 tests)

| Test | Validates |
|------|-----------|
| `test_returns_retriever_types` | `GET /api/options` returns retriever types including "bm25" |
| `test_returns_strategies` | Returns strategies including "merged_pool" |
| `test_returns_defaults` | Returns defaults with `retriever_type` and `k` |

### TestChatEndpoint (8 tests)

| Test | Validates |
|------|-----------|
| `test_normal_answer` | `POST /api/chat` returns non-abstaining answer |
| `test_abstention_at_high_threshold` | High threshold triggers abstention |
| `test_empty_query_returns_400` | Whitespace-only query returns HTTP 400 |
| `test_sources_included` | Response includes `sources` list |
| `test_confidence_in_response` | Response includes `confidence` float |
| `test_config_in_response` | Response includes `config` with retriever type |
| `test_invalid_retriever_type_returns_422` | Invalid retriever type returns HTTP 422 |
| `test_query_whitespace_trimmed` | Leading/trailing whitespace in query is trimmed |

---

## test_models.py — Pydantic Models and Config

### TestServerConfig (5 tests)

| Test | Validates |
|------|-----------|
| `test_default_config_values` | Default host, port, model, k, retriever type, strategy |
| `test_default_knowledge_bases` | Default KBs: ["stpo", "faq-stpo", "faq-ao"] |
| `test_chunk_paths_match_expected_kbs` | `CHUNK_PATHS` keys match expected KBs |
| `test_normalization_map_covers_retriever_types` | Every retriever type has a normalization |
| `test_default_thresholds_cover_normalizations` | Every normalization has a default threshold |

### TestChatRequest (5 tests)

| Test | Validates |
|------|-----------|
| `test_defaults_applied` | Default values for all optional fields |
| `test_query_required` | Missing query raises `ValidationError` |
| `test_query_min_length` | Empty query raises `ValidationError` |
| `test_k_range_validation` | k must be in [1, 50] |
| `test_threshold_range_validation` | threshold must be in [0.0, 1.0] |

### TestChatResponse (3 tests)

| Test | Validates |
|------|-----------|
| `test_full_response` | Complete response with answer, sources, config |
| `test_abstention_response` | Abstention response with level and reason |
| `test_sources_as_list` | Sources field is a list of `SourceReference` |

### TestOptionsResponse (2 tests)

| Test | Validates |
|------|-----------|
| `test_fields_present` | All expected fields are present |
| `test_defaults_dict` | Defaults dict contains expected keys |

---

## test_pipeline.py — Abstention Pipeline Orchestration

Tests for `run_with_abstention()` using `FixedRetriever` and `StubGenerator` stubs.

### TestRunWithAbstention (10 tests)

| Test | Validates |
|------|-----------|
| `test_level1_abstention_low_scores` | All scores below threshold -> Level 1 abstention |
| `test_level2_abstention_llm_abstains` | LLM returns `ABSTENTION:` -> Level 2 abstention |
| `test_normal_answer` | Both levels pass -> normal answer |
| `test_confidence_computed_correctly` | Confidence equals top-1 normalized score |
| `test_filtered_results_passed_to_generator` | Only above-threshold chunks reach generator |
| `test_threshold_zero_no_filtering` | Threshold=0 keeps all results |
| `test_threshold_one_always_abstains` | Threshold=1.0 filters everything |
| `test_bm25_normalization_strategy` | BM25 normalization transforms scores before threshold |
| `test_empty_retrieval_results` | No results -> Level 1 abstention with confidence=0.0 |
| `test_result_fields_complete` | All `AbstentionResult` fields are populated |

---

## test_service.py — PipelineService

Tests for `PipelineService` caching, retriever creation, and chat orchestration.
Uses `KeywordRetriever`, `StubGenerator`, and `@patch` for service internals.

### TestPipelineServiceCaching (3 tests)

| Test | Validates |
|------|-----------|
| `test_same_config_returns_same_instance` | Same (type, KBs, strategy) returns cached retriever |
| `test_different_config_creates_different_instance` | Different config creates new retriever |
| `test_cached_retriever_count` | `cached_retriever_count` tracks cache size correctly |

### TestPipelineServiceRetriever (3 tests)

| Test | Validates |
|------|-----------|
| `test_merged_pool_indexes_merged_chunks` | merged_pool merges chunks from multiple KBs |
| `test_fusion_creates_fusion_retriever` | fusion strategy creates `FusionRetriever` |
| `test_unknown_retriever_type_raises` | Unknown retriever type raises `ValueError` |

### TestPipelineServiceChat (3 tests)

| Test | Validates |
|------|-----------|
| `test_chat_returns_complete_response` | `chat()` returns dict with answer, sources, confidence, config |
| `test_chat_abstention_at_high_threshold` | High threshold triggers abstention |
| `test_chat_normal_answer_at_low_threshold` | Low threshold returns normal answer |

### TestPipelineServiceMisc (1 test)

| Test | Validates |
|------|-----------|
| `test_generator_model` | `generator_model` returns the configured model name |
