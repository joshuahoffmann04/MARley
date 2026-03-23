"""Tests for FastAPI API endpoints using TestClient."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.marley.server.app import create_app
from src.marley.server.config import ServerConfig
from src.marley.server.service import PipelineService
from tests.conftest import KeywordRetriever, StubGenerator


# -- Fixtures --------------------------------------------------------------

CORPUS = [
    {"chunk_id": "c1", "text": "The study period is four semesters", "metadata": {}},
    {"chunk_id": "c2", "text": "Students must complete a thesis", "metadata": {}},
    {"chunk_id": "c3", "text": "The program includes a seminar", "metadata": {}},
]


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with stubbed service."""
    config = ServerConfig(ollama_model="stub-model")

    with (
        patch.object(
            PipelineService, "_create_retriever",
            side_effect=lambda *a, **kw: KeywordRetriever(),
        ),
        patch(
            "src.marley.server.service.load_chunks",
            return_value=CORPUS,
        ),
        patch(
            "src.marley.server.service.OllamaGenerator",
            return_value=StubGenerator(),
        ),
        patch(
            "src.marley.server.app.check_ollama",
            return_value={"available": True, "status_code": 200},
        ),
    ):
        app = create_app(config)
        yield TestClient(app)


# -- Chat Page tests -------------------------------------------------------


class TestChatPage:
    """Tests for GET / (chat page)."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]
        assert "MARley" in resp.text


# -- Debug Page tests ------------------------------------------------------


class TestDebugPage:
    """Tests for GET /debug (debug page)."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/debug")
        assert resp.status_code == 200

    def test_returns_html(self, client: TestClient) -> None:
        resp = client.get("/debug")
        assert "text/html" in resp.headers["content-type"]
        assert "Debug" in resp.text


# -- Health endpoint tests -------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_returns_status(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_includes_ollama_field(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        data = resp.json()
        assert "ollama" in data
        assert data["ollama"] in ("connected", "unavailable")


# -- Options endpoint tests ------------------------------------------------


class TestOptionsEndpoint:
    """Tests for GET /api/options."""

    def test_returns_retriever_types(self, client: TestClient) -> None:
        resp = client.get("/api/options")
        assert resp.status_code == 200
        data = resp.json()
        assert "retriever_types" in data
        assert "bm25" in data["retriever_types"]

    def test_returns_strategies(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        assert "strategies" in data
        assert "merged_pool" in data["strategies"]

    def test_returns_defaults(self, client: TestClient) -> None:
        data = client.get("/api/options").json()
        assert "defaults" in data
        assert "retriever_type" in data["defaults"]
        assert "k" in data["defaults"]


# -- Chat endpoint tests ---------------------------------------------------


class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def test_normal_answer(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "How long is the study period?",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
            "strategy": "merged_pool",
            "threshold": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is False
        assert data["answer"] != ""

    def test_abstention_at_high_threshold(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "How long is the study period?",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
            "strategy": "merged_pool",
            "threshold": 0.99,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is True

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "   ",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
        })
        assert resp.status_code == 400

    def test_sources_included(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "How long is the study period?",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
            "threshold": 0.0,
        })
        data = resp.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_confidence_in_response(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "How long is the study period?",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
            "threshold": 0.0,
        })
        data = resp.json()
        assert "confidence" in data
        assert isinstance(data["confidence"], float)

    def test_config_in_response(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "How long is the study period?",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
            "threshold": 0.0,
        })
        data = resp.json()
        assert "config" in data
        assert data["config"]["retriever_type"] == "bm25"

    def test_invalid_retriever_type_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "test",
            "retriever_type": "invalid",
            "knowledge_bases": ["stpo"],
        })
        assert resp.status_code == 422

    def test_query_whitespace_trimmed(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={
            "query": "  How long is the study period?  ",
            "retriever_type": "bm25",
            "knowledge_bases": ["stpo"],
            "threshold": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] != ""
