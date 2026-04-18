"""Shared fixtures for evaluation tests.

Re-exports canonical stubs from tests/conftest.py so evaluation
tests have a clean import path without depending on the tests/ package.
"""

from unittest.mock import MagicMock

import pytest

from evaluation.judge import Judge
from tests.conftest import (  # noqa: F401
    SMALL_CORPUS,
    FixedRetriever,
    KeywordRetriever,
    StubGenerator,
)


def fake_ragas_scores(raw_results, judge):
    """Stub for ``_score_with_ragas`` returning fixed RAGAS scores.

    Returns one score dict per raw result with deterministic values,
    avoiding any external service calls (Ollama, OpenAI, RAGAS).
    Matches the post-refactor signature (``raw_results, judge``).
    """
    return [
        {"faithfulness": 0.9, "answer_relevancy": 0.85, "factual_correctness": 0.8}
        for _ in raw_results
    ]


@pytest.fixture
def stub_judge() -> Judge:
    """A trivial :class:`Judge` for unit tests that never actually score.

    The ``llm`` and ``embeddings`` slots hold MagicMocks because the
    evaluation flow goes through ``_score_with_ragas``, which callers
    typically monkeypatch via :func:`fake_ragas_scores`.
    """
    return Judge(
        llm=MagicMock(name="stub_llm"),
        embeddings=MagicMock(name="stub_embeddings"),
        batch_size=20,
    )
