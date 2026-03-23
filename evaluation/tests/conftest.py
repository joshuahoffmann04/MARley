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
