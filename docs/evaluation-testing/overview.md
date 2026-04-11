# Evaluation Testing Overview

> Test strategy and architecture for the MARley evaluation pipeline tests.

## Test Structure

```
evaluation/tests/
├── conftest.py                      # Shared fixtures (re-exports + RAGAS stubs)
├── test_utils.py                    # Tests for evaluation/utils.py
├── retrieval/
│   ├── test_metrics.py              # 5 metric functions + evaluate_retriever()
│   ├── test_evaluate.py             # Single-KB evaluation runner
│   ├── test_combined.py             # Merged pool + Fusion evaluation
│   └── test_rrf_tuning.py           # k_rrf sweep (Hybrid + Fusion)
├── generation/
│   ├── test_metrics.py              # GenerationEvalResult + GenerationMetrics
│   ├── test_evaluate.py             # Single-KB generation evaluation + RAGAS stubs
│   └── test_combined.py             # Combined-KB generation evaluation
├── abstention/
│   ├── test_metrics.py              # AbstentionMetrics + compute_abstention_metrics()
│   └── test_evaluate.py             # Level 1 sweep + full two-level evaluation
└── end_to_end/
    ├── test_config.py               # E2EConfig + generate_all_configs()
    ├── test_evaluate.py             # sweep_threshold + run_e2e_config + run_and_report
    └── test_metrics.py              # E2EConfigMetrics + build_comparison_table
```

## Fixture Architecture

### Shared Fixtures (`evaluation/tests/conftest.py`)

Re-exports canonical stubs from the main `tests/conftest.py`:

| Fixture/Stub | Purpose |
|---|---|
| `SMALL_CORPUS` | Minimal chunk corpus for testing |
| `FixedRetriever` | Returns predefined results regardless of query |
| `KeywordRetriever` | BM25-like keyword matching (real retrieval behavior) |
| `StubGenerator` | Returns fixed answers, supports abstention simulation |

### RAGAS Stub (`fake_ragas_scores`)

```python
def fake_ragas_scores(raw_results, ollama_model, ollama_url):
    return [
        {"faithfulness": 0.9, "answer_relevancy": 0.85, "factual_correctness": 0.8}
        for _ in raw_results
    ]
```

This stub replaces `_score_with_ragas()` via `@patch` in generation tests, eliminating the need for Ollama or RAGAS dependencies during testing.

### StubGenerator

The `StubGenerator` (from `tests/conftest.py`) supports:
- Fixed answer text: `StubGenerator(answer="The answer is 42.")`
- Model name: `StubGenerator(model="stub-model")`
- Keyword-based abstention: `StubGenerator(abstain_keywords={"thesis"})` — generates `ABSTENTION: ...` when query contains the keyword

### KeywordRetriever

A lightweight BM25-like retriever that:
- Indexes a real corpus
- Returns chunks with keyword overlap with the query
- Produces realistic scores for normalization testing
- Configurable via `score_multiplier` parameter

## Test Patterns

### Monkeypatching RAGAS

All generation evaluation tests use:

```python
@patch("evaluation.generation.evaluate._score_with_ragas", fake_ragas_scores)
class TestRunGenerationEvaluation:
    ...
```

This patches the RAGAS scoring at the module level, so all generation tests run without external dependencies.

### Temporary JSON Files

Tests that need evaluation data or chunk files write temporary JSON files using `tmp_path`:

```python
def _write_eval_json(path, questions):
    data = {"metadata": {...}, "questions": questions}
    path.write_text(json.dumps(data), encoding="utf-8")
```

### Stub Retriever Patterns

Different test modules use different stub strategies:
- **`_StubRetriever`** (in `test_evaluate.py`): Returns predefined results per query
- **`_StubRetriever`** (in `test_combined.py`): Returns first k indexed chunks
- **`KeywordRetriever`** (shared): Real keyword matching for integration-like tests

## Running Tests

```bash
# All evaluation tests
python -m pytest evaluation/tests/ -v

# Specific module
python -m pytest evaluation/tests/retrieval/test_metrics.py -v

# By component
python -m pytest evaluation/tests/generation/ -v
python -m pytest evaluation/tests/abstention/ -v
python -m pytest evaluation/tests/end_to_end/ -v
```

## Related Documentation

- [Retrieval Tests](retrieval.md)
- [Generation Tests](generation.md)
- [Abstention Tests](abstention.md)
- [End-to-End Tests](end-to-end.md)
- [Utility Tests](utils.md)
