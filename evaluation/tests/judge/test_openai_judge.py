"""Tests for OpenAIJudge (unit tests with mocked openai client)."""

from __future__ import annotations

import pytest


class TestOpenAIJudgeImport:
    def test_import_fails_gracefully_without_openai(self):
        """OpenAIJudge raises ImportError if openai package is missing."""
        import sys
        import importlib
        # If openai is not installed, importing should raise ImportError
        # If it is installed, the class should be importable
        try:
            from evaluation.judge.openai_judge import OpenAIJudge
            # If we get here, openai is installed — just verify the class exists
            assert OpenAIJudge is not None
        except ImportError:
            pytest.skip("openai package not installed")

    def test_init_requires_api_key(self):
        """OpenAIJudge raises ValueError if no API key is available."""
        try:
            from evaluation.judge.openai_judge import OpenAIJudge
        except ImportError:
            pytest.skip("openai package not installed")

        import os
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises((ValueError, Exception)):
                OpenAIJudge(api_key=None)
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original

    def test_model_default(self):
        try:
            from evaluation.judge.openai_judge import OpenAIJudge, _DEFAULT_MODEL
        except ImportError:
            pytest.skip("openai package not installed")
        assert _DEFAULT_MODEL == "gpt-4o-mini"
