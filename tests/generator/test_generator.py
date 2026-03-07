"""Tests for the MARley generator module.

Covers prompt formatting, context assembly, and OllamaGenerator
with a mocked Ollama client to avoid requiring a running server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.marley.generator.base import Generator
from src.marley.generator.ollama import OllamaGenerator
from src.marley.generator.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    format_context,
)
from src.marley.models.generation import GenerationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"chunk_id": "par-7-txt-1", "text": "The standard study period is 4 semesters."},
    {"chunk_id": "par-23-txt-1", "text": "The master thesis has 30 credits."},
]


# ---------------------------------------------------------------------------
# TestFormatContext
# ---------------------------------------------------------------------------


class TestFormatContext:
    """Tests for format_context()."""

    def test_empty_list(self):
        assert format_context([]) == "No context provided."

    def test_single_chunk(self):
        result = format_context([SAMPLE_CHUNKS[0]])
        assert result.startswith("[1]")
        assert "4 semesters" in result

    def test_multiple_chunks_numbered(self):
        result = format_context(SAMPLE_CHUNKS)
        assert "[1]" in result
        assert "[2]" in result
        assert "4 semesters" in result
        assert "30 credits" in result

    def test_preserves_chunk_order(self):
        result = format_context(SAMPLE_CHUNKS)
        pos_1 = result.index("4 semesters")
        pos_2 = result.index("30 credits")
        assert pos_1 < pos_2


# ---------------------------------------------------------------------------
# TestBuildMessages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Tests for build_messages()."""

    def test_returns_two_messages(self):
        msgs = build_messages("How long?", SAMPLE_CHUNKS)
        assert len(msgs) == 2

    def test_system_message_first(self):
        msgs = build_messages("How long?", SAMPLE_CHUNKS)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_user_message_contains_query(self):
        msgs = build_messages("How long?", SAMPLE_CHUNKS)
        assert "How long?" in msgs[1]["content"]

    def test_user_message_contains_context(self):
        msgs = build_messages("How long?", SAMPLE_CHUNKS)
        assert "4 semesters" in msgs[1]["content"]

    def test_empty_context(self):
        msgs = build_messages("How long?", [])
        assert "No context provided." in msgs[1]["content"]


# ---------------------------------------------------------------------------
# TestOllamaGeneratorUnit
# ---------------------------------------------------------------------------


def _mock_chat_response(content="The answer is 4 semesters."):
    """Create a mock Ollama ChatResponse."""
    response = MagicMock()
    response.message.content = content
    response.model = "llama3.1:8b"
    response.prompt_eval_count = 120
    response.eval_count = 25
    return response


class TestOllamaGeneratorUnit:
    """Unit tests for OllamaGenerator with mocked Ollama client."""

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_implements_generator_interface(self, mock_client_cls):
        gen = OllamaGenerator()
        assert isinstance(gen, Generator)

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_returns_generation_result(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response()

        gen = OllamaGenerator()
        result = gen.generate("How long?", SAMPLE_CHUNKS)
        assert isinstance(result, GenerationResult)

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_answer_content(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response("4 semesters.")

        gen = OllamaGenerator()
        result = gen.generate("How long?", SAMPLE_CHUNKS)
        assert result.answer == "4 semesters."

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_records_model(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response()

        gen = OllamaGenerator()
        result = gen.generate("How long?", SAMPLE_CHUNKS)
        assert result.model == "llama3.1:8b"

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_records_chunk_ids(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response()

        gen = OllamaGenerator()
        result = gen.generate("How long?", SAMPLE_CHUNKS)
        assert result.context_chunk_ids == ["par-7-txt-1", "par-23-txt-1"]

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_records_token_counts(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response()

        gen = OllamaGenerator()
        result = gen.generate("How long?", SAMPLE_CHUNKS)
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 25

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_empty_context(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response("I cannot answer.")

        gen = OllamaGenerator()
        result = gen.generate("How long?", [])
        assert result.context_chunk_ids == []
        assert result.answer == "I cannot answer."

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_generate_strips_whitespace(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response("  answer  \n")

        gen = OllamaGenerator()
        result = gen.generate("q?", [])
        assert result.answer == "answer"

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_custom_model(self, mock_client_cls):
        gen = OllamaGenerator(model="mistral:7b")
        assert gen.model == "mistral:7b"

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_chat_called_with_correct_model(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = _mock_chat_response()

        gen = OllamaGenerator(model="mistral:7b")
        gen.generate("q?", SAMPLE_CHUNKS)

        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["model"] == "mistral:7b"

    @patch("src.marley.generator.ollama.ollama_lib.Client")
    def test_none_token_counts_default_to_zero(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        resp = _mock_chat_response()
        resp.prompt_eval_count = None
        resp.eval_count = None
        mock_client.chat.return_value = resp

        gen = OllamaGenerator()
        result = gen.generate("q?", [])
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


# ---------------------------------------------------------------------------
# Integration tests (require running Ollama server)
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama server not running")
class TestOllamaGeneratorIntegration:
    """Integration tests against a live Ollama server."""

    def test_generate_returns_nonempty_answer(self):
        gen = OllamaGenerator()
        result = gen.generate(
            "How long is the standard study period?",
            SAMPLE_CHUNKS,
        )
        assert len(result.answer) > 0

    def test_generate_returns_generation_result_type(self):
        gen = OllamaGenerator()
        result = gen.generate("How many credits?", SAMPLE_CHUNKS)
        assert isinstance(result, GenerationResult)

    def test_token_counts_positive(self):
        gen = OllamaGenerator()
        result = gen.generate("How long?", SAMPLE_CHUNKS)
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
