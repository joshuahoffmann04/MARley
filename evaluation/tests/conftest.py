"""Shared fixtures for evaluation tests.

Re-exports canonical stubs from tests/conftest.py so evaluation
tests have a clean import path without depending on the tests/ package.
"""

from tests.conftest import (  # noqa: F401
    SMALL_CORPUS,
    FixedRetriever,
    KeywordRetriever,
    StubGenerator,
)


def fake_ragas_scores(raw_results, ollama_model, ollama_url):
    """Stub for _score_with_ragas returning fixed RAGAS scores.

    Returns one score dict per raw result with deterministic values,
    avoiding any external service calls (Ollama, RAGAS).
    """
    return [
        {"faithfulness": 0.9, "answer_relevancy": 0.85, "factual_correctness": 0.8}
        for _ in raw_results
    ]
