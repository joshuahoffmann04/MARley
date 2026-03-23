"""Tests for RRF k-parameter tuning module."""

from __future__ import annotations

import json

import pytest

from evaluation.retrieval.rrf_tuning import (
    DEFAULT_SWEEP_VALUES,
    sweep_fusion_k_rrf,
    sweep_hybrid_k_rrf,
)
from tests.conftest import KeywordRetriever


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _write_chunks(path, chunks):
    data = {"chunks": chunks}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_eval(path, questions):
    data = {"metadata": {"version": "1.0"}, "questions": questions}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


CHUNKS = [
    {"chunk_id": "c1", "text": "The study period is four semesters.", "metadata": {}},
    {"chunk_id": "c2", "text": "Students must complete a thesis.", "metadata": {}},
    {"chunk_id": "c3", "text": "The program includes a seminar.", "metadata": {}},
]

QUESTIONS = [
    {
        "id": "q1",
        "question": "How long is the study period?",
        "relevant_chunks": ["c1"],
        "category": "direct",
        "expected_abstention": False,
    },
    {
        "id": "q2",
        "question": "Is a thesis required for students?",
        "relevant_chunks": ["c2"],
        "category": "direct",
        "expected_abstention": False,
    },
]


# ---------------------------------------------------------------------------
# Hybrid sweep tests
# ---------------------------------------------------------------------------


class TestSweepHybridKRRF:
    @pytest.fixture()
    def setup(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, CHUNKS)
        _write_eval(ep, QUESTIONS)
        return cp, ep

    def test_returns_valid_structure(self, setup):
        cp, ep = setup
        result = sweep_hybrid_k_rrf(
            retriever_factory=lambda k_rrf: KeywordRetriever(),
            chunk_path=cp,
            eval_path=ep,
            sweep_values=[10, 60],
        )
        assert "best_k_rrf" in result
        assert "best_metrics" in result
        assert "sweep_results" in result
        assert "config" in result

    def test_sweep_covers_all_values(self, setup):
        cp, ep = setup
        values = [1, 10, 60, 100]
        result = sweep_hybrid_k_rrf(
            retriever_factory=lambda k_rrf: KeywordRetriever(),
            chunk_path=cp,
            eval_path=ep,
            sweep_values=values,
        )
        assert len(result["sweep_results"]) == len(values)
        actual_k_values = [r["k_rrf"] for r in result["sweep_results"]]
        assert actual_k_values == values

    def test_best_k_rrf_in_sweep_range(self, setup):
        cp, ep = setup
        values = [5, 30, 60]
        result = sweep_hybrid_k_rrf(
            retriever_factory=lambda k_rrf: KeywordRetriever(),
            chunk_path=cp,
            eval_path=ep,
            sweep_values=values,
        )
        assert result["best_k_rrf"] in values

    def test_metrics_valid(self, setup):
        cp, ep = setup
        result = sweep_hybrid_k_rrf(
            retriever_factory=lambda k_rrf: KeywordRetriever(),
            chunk_path=cp,
            eval_path=ep,
            sweep_values=[60],
        )
        m = result["sweep_results"][0]["metrics"]
        assert "precision_at_k" in m
        assert "recall_at_k" in m
        assert "mrr" in m
        assert 0.0 <= m["precision_at_k"] <= 1.0
        assert 0.0 <= m["recall_at_k"] <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0

    def test_default_sweep_values_used(self, setup):
        cp, ep = setup
        result = sweep_hybrid_k_rrf(
            retriever_factory=lambda k_rrf: KeywordRetriever(),
            chunk_path=cp,
            eval_path=ep,
        )
        assert len(result["sweep_results"]) == len(DEFAULT_SWEEP_VALUES)


# ---------------------------------------------------------------------------
# Fusion sweep tests
# ---------------------------------------------------------------------------


class TestSweepFusionKRRF:
    @pytest.fixture()
    def setup(self, tmp_path):
        # Two KBs with distinct chunks
        cp1 = tmp_path / "kb1-chunks.json"
        cp2 = tmp_path / "kb2-chunks.json"
        _write_chunks(cp1, [
            {"chunk_id": "a1", "text": "The study period is four semesters.", "metadata": {}},
            {"chunk_id": "a2", "text": "Students must complete a thesis.", "metadata": {}},
        ])
        _write_chunks(cp2, [
            {"chunk_id": "b1", "text": "The program lasts four semesters total.", "metadata": {}},
            {"chunk_id": "b2", "text": "A seminar is part of the curriculum.", "metadata": {}},
        ])
        ep1 = tmp_path / "eval-kb1.json"
        ep2 = tmp_path / "eval-kb2.json"
        _write_eval(ep1, [
            {"id": "q1", "question": "How long is the study period?",
             "relevant_chunks": ["a1"], "category": "direct",
             "expected_abstention": False},
        ])
        _write_eval(ep2, [
            {"id": "q1", "question": "How long is the study period?",
             "relevant_chunks": ["b1"], "category": "direct",
             "expected_abstention": False},
        ])
        return (
            {"kb1": cp1, "kb2": cp2},
            {"kb1": ep1, "kb2": ep2},
        )

    def test_returns_valid_structure(self, setup):
        chunk_paths, eval_paths = setup
        result = sweep_fusion_k_rrf(
            retriever_factory=KeywordRetriever,
            chunk_paths=chunk_paths,
            eval_paths=eval_paths,
            sweep_values=[10, 60],
        )
        assert "best_k_rrf" in result
        assert "sweep_results" in result
        assert result["sweep_type"] == "fusion"

    def test_sweep_covers_all_values(self, setup):
        chunk_paths, eval_paths = setup
        values = [1, 30, 60]
        result = sweep_fusion_k_rrf(
            retriever_factory=KeywordRetriever,
            chunk_paths=chunk_paths,
            eval_paths=eval_paths,
            sweep_values=values,
        )
        assert len(result["sweep_results"]) == len(values)

    def test_best_k_rrf_in_sweep_range(self, setup):
        chunk_paths, eval_paths = setup
        values = [5, 60, 100]
        result = sweep_fusion_k_rrf(
            retriever_factory=KeywordRetriever,
            chunk_paths=chunk_paths,
            eval_paths=eval_paths,
            sweep_values=values,
        )
        assert result["best_k_rrf"] in values

    def test_config_records_knowledge_bases(self, setup):
        chunk_paths, eval_paths = setup
        result = sweep_fusion_k_rrf(
            retriever_factory=KeywordRetriever,
            chunk_paths=chunk_paths,
            eval_paths=eval_paths,
            sweep_values=[60],
        )
        assert sorted(result["config"]["knowledge_bases"]) == ["kb1", "kb2"]
        assert result["config"]["total_chunks"] == 4

    def test_edge_case_k_rrf_one(self, setup):
        chunk_paths, eval_paths = setup
        result = sweep_fusion_k_rrf(
            retriever_factory=KeywordRetriever,
            chunk_paths=chunk_paths,
            eval_paths=eval_paths,
            sweep_values=[1],
        )
        m = result["sweep_results"][0]["metrics"]
        assert 0.0 <= m["recall_at_k"] <= 1.0
