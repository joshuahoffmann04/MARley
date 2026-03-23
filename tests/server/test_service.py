"""Tests for PipelineService with stubbed retrievers and generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.marley.retrieval.fusion import FusionRetriever
from src.marley.server.config import ServerConfig
from src.marley.server.service import PipelineService
from tests.conftest import KeywordRetriever, StubGenerator


# -- Fixtures --------------------------------------------------------------


CORPUS = [
    {"chunk_id": "c1", "text": "The study period is four semesters", "metadata": {}},
    {"chunk_id": "c2", "text": "Students must complete a thesis", "metadata": {}},
    {"chunk_id": "c3", "text": "The program includes a seminar", "metadata": {}},
]


def _make_service() -> PipelineService:
    """Create a PipelineService with stubbed retriever/generator."""
    config = ServerConfig(ollama_model="stub-model")
    service = PipelineService(config)
    # Replace generator with stub
    service._generator = StubGenerator()
    return service


def _patch_retriever_creation():
    """Patch _create_retriever and load_chunks to use stubs."""
    return [
        patch.object(
            PipelineService,
            "_create_retriever",
            return_value=KeywordRetriever(),
        ),
        patch(
            "src.marley.server.service.load_chunks",
            return_value=CORPUS,
        ),
    ]


# -- Tests -----------------------------------------------------------------


class TestPipelineServiceCaching:
    """Tests for retriever caching behavior."""

    def test_same_config_returns_same_instance(self) -> None:
        service = _make_service()
        patches = _patch_retriever_creation()
        for p in patches:
            p.start()
        try:
            r1 = service.get_retriever("bm25", ["stpo"], "merged_pool")
            r2 = service.get_retriever("bm25", ["stpo"], "merged_pool")
            assert r1 is r2
        finally:
            for p in patches:
                p.stop()

    def test_different_config_creates_different_instance(self) -> None:
        service = _make_service()
        with (
            patch.object(
                PipelineService, "_create_retriever",
                side_effect=lambda *a, **kw: KeywordRetriever(),
            ),
            patch(
                "src.marley.server.service.load_chunks",
                return_value=CORPUS,
            ),
        ):
            r1 = service.get_retriever("bm25", ["stpo"], "merged_pool")
            r2 = service.get_retriever("bm25", ["faq-stpo"], "merged_pool")
            assert r1 is not r2

    def test_cached_retriever_count(self) -> None:
        service = _make_service()
        assert service.cached_retriever_count == 0
        patches = _patch_retriever_creation()
        for p in patches:
            p.start()
        try:
            service.get_retriever("bm25", ["stpo"], "merged_pool")
            assert service.cached_retriever_count == 1
            service.get_retriever("bm25", ["stpo"], "merged_pool")
            assert service.cached_retriever_count == 1
            service.get_retriever("bm25", ["faq-stpo"], "merged_pool")
            assert service.cached_retriever_count == 2
        finally:
            for p in patches:
                p.stop()


class TestPipelineServiceRetriever:
    """Tests for retriever creation strategies."""

    def test_merged_pool_indexes_merged_chunks(self) -> None:
        service = _make_service()
        created_retriever = KeywordRetriever()
        with (
            patch.object(
                PipelineService, "_create_retriever", return_value=created_retriever,
            ),
            patch(
                "src.marley.server.service.load_chunks",
                return_value=CORPUS,
            ),
        ):
            r = service.get_retriever("bm25", ["stpo", "faq-stpo"], "merged_pool")
            # Merged 2 KBs x 3 chunks each = 6 total
            assert r.size == 6

    def test_fusion_creates_fusion_retriever(self) -> None:
        service = _make_service()
        with (
            patch.object(
                PipelineService, "_create_retriever",
                side_effect=lambda *a, **kw: KeywordRetriever(),
            ),
            patch(
                "src.marley.server.service.load_chunks",
                return_value=CORPUS,
            ),
        ):
            r = service.get_retriever("bm25", ["stpo", "faq-stpo"], "fusion")
            assert isinstance(r, FusionRetriever)

    def test_unknown_retriever_type_raises(self) -> None:
        service = _make_service()
        with pytest.raises(ValueError, match="Unknown retriever type"):
            service._create_retriever("invalid")


class TestPipelineServiceChat:
    """Tests for the chat method."""

    def test_chat_returns_complete_response(self) -> None:
        service = _make_service()
        with (
            patch.object(
                PipelineService, "_create_retriever",
                side_effect=lambda *a, **kw: KeywordRetriever(),
            ),
            patch(
                "src.marley.server.service.load_chunks",
                return_value=CORPUS,
            ),
        ):
            result = service.chat(
                "How long is the study period?",
                retriever_type="bm25",
                knowledge_bases=["stpo"],
                strategy="merged_pool",
            )
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            assert "config" in result
            assert result["config"]["retriever_type"] == "bm25"

    def test_chat_abstention_at_high_threshold(self) -> None:
        service = _make_service()
        with (
            patch.object(
                PipelineService, "_create_retriever",
                side_effect=lambda *a, **kw: KeywordRetriever(),
            ),
            patch(
                "src.marley.server.service.load_chunks",
                return_value=CORPUS,
            ),
        ):
            result = service.chat(
                "How long is the study period?",
                retriever_type="bm25",
                knowledge_bases=["stpo"],
                strategy="merged_pool",
                threshold=0.99,
            )
            assert result["abstained"] is True
            assert result["abstention_level"] == 1

    def test_chat_normal_answer_at_low_threshold(self) -> None:
        service = _make_service()
        with (
            patch.object(
                PipelineService, "_create_retriever",
                side_effect=lambda *a, **kw: KeywordRetriever(),
            ),
            patch(
                "src.marley.server.service.load_chunks",
                return_value=CORPUS,
            ),
        ):
            result = service.chat(
                "How long is the study period?",
                retriever_type="bm25",
                knowledge_bases=["stpo"],
                strategy="merged_pool",
                threshold=0.0,
            )
            assert result["abstained"] is False
            assert result["answer"] != ""


class TestPipelineServiceMisc:
    """Miscellaneous service tests."""

    def test_generator_model(self) -> None:
        service = _make_service()
        assert service.generator_model == "stub-model"
