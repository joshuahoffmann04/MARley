# Generator Test Documentation

**Test file:** `tests/generator/test_generator.py`
**Total tests:** 23
**Run command:** `python -m pytest tests/generator/test_generator.py -v`

---

## Test Structure

### Unit Tests (no external dependencies)

| Class | Tests | What is verified |
|---|---|---|
| `TestFormatContext` | 4 | Empty list fallback, single/multiple chunk numbering, chunk order preservation. |
| `TestBuildMessages` | 5 | Message count, system prompt present, query in user message, context in user message, empty context handling. |
| `TestOllamaGeneratorUnit` | 11 | Interface implementation, return type, answer content, model recording, chunk ID tracking, token counts, empty context, whitespace stripping, custom model, correct API call, None token defaults. |

### Integration Tests (require running Ollama server)

| Class | Tests | What is verified |
|---|---|---|
| `TestOllamaGeneratorIntegration` | 3 | Non-empty answer, correct return type, positive token counts. |

---

## Test Strategy

Unit tests mock the Ollama client via `unittest.mock.patch`, allowing full coverage of the generator logic without a running server. Integration tests are guarded by a connectivity check and skipped automatically when Ollama is unavailable.

---

## CI Considerations

- All unit tests (20) run without external dependencies.
- Integration tests (3) are skipped if Ollama is not reachable.
