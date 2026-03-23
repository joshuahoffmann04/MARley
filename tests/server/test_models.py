"""Tests for server Pydantic models and configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.marley.server.config import (
    CHUNK_PATHS,
    DEFAULT_THRESHOLDS,
    NORMALIZATION_MAP,
    RETRIEVER_TYPES,
    STRATEGIES,
    ServerConfig,
)
from src.marley.server.models import (
    ChatConfigInfo,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    OptionsResponse,
    SourceReference,
)


# -- ServerConfig tests ----------------------------------------------------


class TestServerConfig:
    """Tests for ServerConfig defaults and validation."""

    def test_default_config_values(self) -> None:
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.ollama_model == "llama3.1:latest"
        assert config.k == 5
        assert config.default_retriever_type == "hybrid"
        assert config.default_strategy == "merged_pool"

    def test_default_knowledge_bases(self) -> None:
        config = ServerConfig()
        assert config.default_knowledge_bases == ["stpo", "faq-stpo", "faq-ao"]

    def test_chunk_paths_match_expected_kbs(self) -> None:
        assert set(CHUNK_PATHS.keys()) == {"stpo", "faq-stpo", "faq-ao"}

    def test_normalization_map_covers_retriever_types(self) -> None:
        for rt in RETRIEVER_TYPES:
            assert rt in NORMALIZATION_MAP, f"Missing normalization for {rt}"

    def test_default_thresholds_cover_normalizations(self) -> None:
        for norm in NORMALIZATION_MAP.values():
            assert norm in DEFAULT_THRESHOLDS, f"Missing threshold for {norm}"


# -- ChatRequest tests -----------------------------------------------------


class TestChatRequest:
    """Tests for ChatRequest validation."""

    def test_defaults_applied(self) -> None:
        req = ChatRequest(query="test question")
        assert req.retriever_type == "hybrid"
        assert req.strategy == "merged_pool"
        assert req.k == 5
        assert req.threshold is None
        assert req.knowledge_bases == ["stpo", "faq-stpo", "faq-ao"]

    def test_query_required(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_query_min_length(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(query="")

    def test_k_range_validation(self) -> None:
        req = ChatRequest(query="test", k=1)
        assert req.k == 1
        with pytest.raises(ValidationError):
            ChatRequest(query="test", k=0)
        with pytest.raises(ValidationError):
            ChatRequest(query="test", k=51)

    def test_threshold_range_validation(self) -> None:
        req = ChatRequest(query="test", threshold=0.5)
        assert req.threshold == 0.5
        with pytest.raises(ValidationError):
            ChatRequest(query="test", threshold=-0.1)
        with pytest.raises(ValidationError):
            ChatRequest(query="test", threshold=1.1)


# -- ChatResponse tests ----------------------------------------------------


class TestChatResponse:
    """Tests for ChatResponse structure."""

    def test_full_response(self) -> None:
        resp = ChatResponse(
            answer="The answer.",
            abstained=False,
            confidence=0.8,
            sources=[
                SourceReference(chunk_id="c1", text="text", score=0.9),
            ],
            config=ChatConfigInfo(
                retriever_type="hybrid",
                knowledge_bases=["stpo"],
                strategy="merged_pool",
                k=5,
                threshold=0.3,
                normalization_strategy="rrf",
                model="llama3.1:latest",
            ),
        )
        assert resp.answer == "The answer."
        assert not resp.abstained
        assert len(resp.sources) == 1

    def test_abstention_response(self) -> None:
        resp = ChatResponse(
            answer="",
            abstained=True,
            abstention_level=1,
            abstention_reason="low confidence",
            confidence=0.1,
            sources=[],
            config=ChatConfigInfo(
                retriever_type="bm25",
                knowledge_bases=["stpo"],
                strategy="single",
                k=5,
                threshold=0.3,
                normalization_strategy="bm25",
                model="llama3.1:latest",
            ),
        )
        assert resp.abstained
        assert resp.abstention_level == 1

    def test_sources_as_list(self) -> None:
        resp = ChatResponse(
            answer="test",
            abstained=False,
            confidence=0.5,
            sources=[
                SourceReference(chunk_id="c1", text="a", score=0.9),
                SourceReference(chunk_id="c2", text="b", score=0.7),
            ],
            config=ChatConfigInfo(
                retriever_type="hybrid",
                knowledge_bases=["stpo"],
                strategy="merged_pool",
                k=5,
                threshold=0.3,
                normalization_strategy="rrf",
                model="test",
            ),
        )
        assert isinstance(resp.sources, list)
        assert len(resp.sources) == 2


# -- OptionsResponse tests -------------------------------------------------


class TestOptionsResponse:
    """Tests for OptionsResponse structure."""

    def test_fields_present(self) -> None:
        resp = OptionsResponse(
            retriever_types=["bm25"],
            knowledge_bases=["stpo"],
            strategies=["single"],
            defaults={"k": 5},
            ollama_model="test",
            ollama_status="connected",
        )
        assert resp.retriever_types == ["bm25"]
        assert resp.ollama_status == "connected"

    def test_defaults_dict(self) -> None:
        resp = OptionsResponse(
            retriever_types=[],
            knowledge_bases=[],
            strategies=[],
            defaults={"retriever_type": "hybrid", "k": 5},
            ollama_model="test",
            ollama_status="unavailable",
        )
        assert "retriever_type" in resp.defaults
        assert resp.defaults["k"] == 5
