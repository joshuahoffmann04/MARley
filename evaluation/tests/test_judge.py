"""Tests for the judge factory module.

These tests exercise constructor arguments only. They do not call the
Ollama server, the OpenAI API, or construct a real CUDA embedding
model; the HuggingFaceEmbeddings class is stubbed so assertions can
run on CPU-only CI machines.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evaluation.judge import Judge, make_judge


@pytest.fixture
def _stub_ragas(monkeypatch):
    """Stub the two RAGAS factory calls used by make_judge().

    Returns the two MagicMocks so individual tests can inspect call args.
    """
    llm_mock = MagicMock(name="llm_factory")
    llm_mock.return_value = MagicMock(name="evaluator_llm")

    emb_mock = MagicMock(name="HuggingFaceEmbeddings")
    emb_mock.return_value = MagicMock(name="embeddings_instance")

    monkeypatch.setattr("ragas.llms.llm_factory", llm_mock)
    monkeypatch.setattr("ragas.embeddings.HuggingFaceEmbeddings", emb_mock)

    # AsyncOpenAI stub — avoids real HTTP client setup.
    client_mock = MagicMock(name="AsyncOpenAI")
    monkeypatch.setattr("openai.AsyncOpenAI", client_mock)

    return {"llm": llm_mock, "embeddings": emb_mock, "client": client_mock}


class TestMakeJudgeOllama:
    """Ollama backend: local server, batch size 20."""

    def test_make_judge_ollama_returns_judge_object(self, _stub_ragas):
        judge = make_judge("ollama")
        assert isinstance(judge, Judge)

    def test_make_judge_ollama_batch_size_is_20(self, _stub_ragas):
        judge = make_judge("ollama")
        assert judge.batch_size == 20

    def test_make_judge_ollama_uses_model_and_url(self, _stub_ragas):
        make_judge(
            "ollama",
            ollama_model="llama3.1:latest",
            ollama_url="http://localhost:11434",
        )
        client_call = _stub_ragas["client"].call_args
        assert client_call.kwargs["base_url"] == "http://localhost:11434/v1"
        assert client_call.kwargs["api_key"] == "ollama"

        llm_call = _stub_ragas["llm"].call_args
        assert llm_call.args[0] == "llama3.1:latest"
        assert llm_call.kwargs["provider"] == "openai"


class TestMakeJudgeOpenAI:
    """OpenAI backend: gpt-4o-mini, batch size 50, requires API key."""

    def test_make_judge_openai_batch_size_is_50(self, _stub_ragas, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        judge = make_judge("openai")
        assert judge.batch_size == 50

    def test_make_judge_openai_uses_gpt4o_mini(self, _stub_ragas, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        make_judge("openai")
        llm_call = _stub_ragas["llm"].call_args
        assert llm_call.args[0] == "gpt-4o-mini"

    def test_make_judge_openai_raises_when_no_api_key(
        self, _stub_ragas, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
            make_judge("openai")


class TestMakeJudgeValidation:
    """Backend-name validation."""

    def test_make_judge_unknown_backend_raises(self, _stub_ragas):
        with pytest.raises(ValueError, match="Unknown judge backend"):
            make_judge("anthropic")  # type: ignore[arg-type]


class TestJudgeEmbeddingsDevice:
    """Embeddings must always be constructed on CUDA."""

    def test_judge_embeddings_use_cuda_device(self, _stub_ragas, monkeypatch):
        # Both backends share the embedding construction path.
        make_judge("ollama")
        emb_call = _stub_ragas["embeddings"].call_args
        assert emb_call.kwargs["device"] == "cuda"
        assert (
            emb_call.kwargs["model"]
            == "sentence-transformers/all-mpnet-base-v2"
        )

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _stub_ragas["embeddings"].reset_mock()
        make_judge("openai")
        emb_call = _stub_ragas["embeddings"].call_args
        assert emb_call.kwargs["device"] == "cuda"
