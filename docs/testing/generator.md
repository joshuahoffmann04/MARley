# Generator Tests

> Tests for LLM generation via Ollama in `src/marley/generator/`.

**Test file**: `tests/generator/test_generator.py`
**Total**: 24 tests (21 unit + 3 integration)

## Unit Tests

### TestFormatContext (4 tests)

Tests for the `format_context()` function that assembles retrieval chunks into
a numbered context string for the LLM prompt.

| Test | Validates |
|------|-----------|
| `test_empty_list` | Returns `"No context provided."` |
| `test_single_chunk` | Formats as `[1] <text>` |
| `test_multiple_chunks_numbered` | Multiple chunks get sequential numbers |
| `test_preserves_chunk_order` | Chunk ordering is maintained |

### TestBuildMessages (5 tests)

Tests for the `build_messages()` function that constructs the chat message list.

| Test | Validates |
|------|-----------|
| `test_returns_two_messages` | Returns exactly 2 messages (system + user) |
| `test_system_message_first` | First message has role `"system"` with `SYSTEM_PROMPT` |
| `test_user_message_contains_query` | User message includes the query |
| `test_user_message_contains_context` | User message includes formatted context |
| `test_empty_context` | Empty context shows `"No context provided."` |

### TestOllamaGeneratorUnit (12 tests)

Tests `OllamaGenerator` with a mocked Ollama client (`@patch`).

| Test | Validates |
|------|-----------|
| `test_implements_generator_interface` | `OllamaGenerator` is a `Generator` instance |
| `test_generate_returns_generation_result` | `generate()` returns `GenerationResult` |
| `test_generate_answer_content` | Answer text matches mock response |
| `test_generate_records_model` | Model name from response is recorded |
| `test_generate_records_chunk_ids` | Context chunk IDs are captured |
| `test_generate_records_token_counts` | Prompt and completion token counts are recorded |
| `test_generate_empty_context` | Empty context results in empty `context_chunk_ids` |
| `test_generate_strips_whitespace` | Answer whitespace is stripped |
| `test_custom_model` | Custom model name is stored |
| `test_model_is_property` | `model` is a `property` descriptor |
| `test_chat_called_with_correct_model` | `client.chat()` receives the configured model name |
| `test_none_token_counts_default_to_zero` | `None` token counts default to 0 |

**Note**: This is the one module that uses `unittest.mock.patch` rather than
stubs, because `OllamaGenerator` wraps an external HTTP client where a stub
approach would require reimplementing the entire Ollama client interface.

## Integration Tests (Require Running Ollama Server)

### TestOllamaGeneratorIntegration (3 tests)

Guarded with `@pytest.mark.integration` and a skipif for Ollama availability.

| Test | Validates |
|------|-----------|
| `test_generate_returns_nonempty_answer` | Live generation produces a non-empty answer |
| `test_generate_returns_generation_result_type` | Result is a `GenerationResult` instance |
| `test_token_counts_positive` | Prompt and completion tokens are positive |
