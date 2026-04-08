# LLM Judge Test Documentation

**Test files:** `evaluation/tests/judge/`
**Total tests:** 46
**Run command:** `python -m pytest evaluation/tests/judge/ -v`

---

## Test Structure

### test_base.py (9 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestJudgementResult` | 2 | All five fields accessible, dataclass field names are correct. |
| `TestJudgeABC` | 7 | Abstract class cannot be instantiated, stub implements interface, model property accessible, model is a property (not attribute), judge returns JudgementResult, question_id propagated, fixed scores passed through correctly. |

### test_prompts.py (15 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestFormatContext` | 4 | Empty input returns fallback text, single chunk numbered `[1]`, multiple chunks all numbered, chunks appear in insertion order. |
| `TestBuildJudgeMessages` | 7 | Returns two messages, system message is first, question in user message, generated answer in user message, reference answer in user message, context in user message, empty context fallback. |
| `TestJudgeSystemPrompt` | 4 | Contains `faithfulness`, `answer_relevance`, `correctness`, and `JSON` instruction. |

### test_ollama_judge.py (19 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestParseScores` | 7 | Valid JSON parsed correctly, scores clamped above 1.0, scores clamped below 0.0, missing key defaults to 0.0, JSON embedded in prose extracted, invalid JSON returns zeros, non-numeric value defaults to 0.0. |
| `TestOllamaJudgeUnit` | 10 | Implements Judge interface, model property returns correct value, model is a property, returns JudgementResult, scores parsed correctly, question_id propagated, model recorded from response, abstained answer returns sentinel (f=1, r=0, c=0), empty answer returns sentinel, abstention check is case-insensitive, chat called with JSON format. |
| `TestOllamaJudgeIntegration` | 2 (integration) | Scores in [0,1] for real answer, high scores for grounded correct answer. |

### test_openai_judge.py (3 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestOpenAIJudgeImport` | 3 | Import fails gracefully without `openai` package, ValueError raised without API key, default model is `gpt-4o-mini`. |

---

## Test Strategy

**Unit tests** mock the Ollama client via `unittest.mock.patch`, allowing all logic
(JSON parsing, sentinel handling, abstention detection) to be tested without a running
server. The `_FixedJudge` stub in `test_base.py` validates the abstract interface.

**Integration tests** in `TestOllamaJudgeIntegration` require Ollama and are
guarded by `@pytest.mark.skipif(not _ollama_available(), ...)`.

**OpenAI tests** are lightweight: they only verify import behaviour, API key
validation, and default model selection — no actual API calls are made.

---

## Fixtures

No shared fixtures — unit tests use `unittest.mock.patch` inline.
