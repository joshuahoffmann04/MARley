"""Tests for the generation judge module.

Covers JSON parsing, fallback heuristics, and OllamaJudge with
mocked Ollama client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evaluation.generation.judge import (
    Judge,
    JudgementResult,
    OllamaJudge,
    _parse_judgement,
)


# ---------------------------------------------------------------------------
# TestParseJudgement
# ---------------------------------------------------------------------------


class TestParseJudgement:
    """Tests for _parse_judgement() JSON parsing and fallbacks."""

    def test_valid_json_correct(self):
        text = '{"correct": true, "confidence": 0.95, "reasoning": "Matches."}'
        result = _parse_judgement(text)
        assert result.correct is True
        assert result.confidence == 0.95
        assert result.reasoning == "Matches."

    def test_valid_json_incorrect(self):
        text = '{"correct": false, "confidence": 0.8, "reasoning": "Wrong."}'
        result = _parse_judgement(text)
        assert result.correct is False

    def test_markdown_code_fence_stripped(self):
        text = '```json\n{"correct": true, "confidence": 0.9, "reasoning": "OK"}\n```'
        result = _parse_judgement(text)
        assert result.correct is True

    def test_fallback_on_invalid_json(self):
        text = "This is not JSON at all"
        result = _parse_judgement(text)
        assert result.correct is False
        assert result.confidence == 0.5

    def test_fallback_detects_correct_true_string(self):
        text = 'The answer has "correct": true in it somewhere'
        result = _parse_judgement(text)
        assert result.correct is True

    def test_missing_fields_use_defaults(self):
        text = '{"correct": true}'
        result = _parse_judgement(text)
        assert result.correct is True
        assert result.confidence == 0.5
        assert result.reasoning == ""


# ---------------------------------------------------------------------------
# TestOllamaJudgeUnit
# ---------------------------------------------------------------------------


class TestOllamaJudgeUnit:
    """Unit tests for OllamaJudge with mocked Ollama client."""

    @patch("evaluation.generation.judge.ollama_lib.Client")
    def test_implements_judge_interface(self, mock_client_cls):
        judge = OllamaJudge()
        assert isinstance(judge, Judge)

    @patch("evaluation.generation.judge.ollama_lib.Client")
    def test_evaluate_returns_judgement_result(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        resp = MagicMock()
        resp.message.content = '{"correct": true, "confidence": 0.9, "reasoning": "Match"}'
        mock_client.chat.return_value = resp

        judge = OllamaJudge()
        result = judge.evaluate("Q?", "ref", "gen")
        assert isinstance(result, JudgementResult)
        assert result.correct is True

    @patch("evaluation.generation.judge.ollama_lib.Client")
    def test_evaluate_passes_json_format(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        resp = MagicMock()
        resp.message.content = '{"correct": false, "confidence": 0.7, "reasoning": "Wrong"}'
        mock_client.chat.return_value = resp

        judge = OllamaJudge()
        judge.evaluate("Q?", "ref", "gen")

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["format"] == "json"

    @patch("evaluation.generation.judge.ollama_lib.Client")
    def test_evaluate_includes_all_inputs(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        resp = MagicMock()
        resp.message.content = '{"correct": true, "confidence": 1.0, "reasoning": "OK"}'
        mock_client.chat.return_value = resp

        judge = OllamaJudge()
        judge.evaluate("My question", "My reference", "My generated")

        call_kwargs = mock_client.chat.call_args.kwargs
        user_msg = call_kwargs["messages"][1]["content"]
        assert "My question" in user_msg
        assert "My reference" in user_msg
        assert "My generated" in user_msg
