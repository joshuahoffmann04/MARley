"""Tests for the combined knowledge base retrieval evaluation module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.retrieval.combined import (
    run_fusion_evaluation,
    run_merged_pool_evaluation,
)
from evaluation.utils import merge_chunks, merge_evaluation_data
from evaluation.retrieval.metrics import RetrievalMetrics
from src.marley.models.retrieval import RetrievalResult, Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_chunks(path: Path, chunks: list[dict]) -> None:
    """Write a minimal chunk JSON file."""
    data = {"chunks": chunks}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_eval(path: Path, questions: list[dict], kb: str = "test") -> None:
    """Write a minimal evaluation JSON file."""
    data = {
        "metadata": {"version": "1.0", "knowledge_base": kb},
        "questions": questions,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _chunk(cid: str, text: str = "") -> dict:
    """Create a minimal chunk dict."""
    return {"chunk_id": cid, "text": text or f"text-{cid}", "metadata": {}}


class _StubRetriever(Retriever):
    """Retriever that returns the first k indexed chunks for any query."""

    def __init__(self) -> None:
        self._corpus: list[dict] = []

    def index(self, corpus: list[dict]) -> None:
        self._corpus = corpus

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        results = []
        for i, doc in enumerate(self._corpus[:k]):
            results.append(RetrievalResult(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                score=1.0 / (i + 1),
                metadata=doc.get("metadata", {}),
            ))
        return results

    @property
    def size(self) -> int:
        return len(self._corpus)


# ---------------------------------------------------------------------------
# Tests: merge_chunks
# ---------------------------------------------------------------------------

class TestMergeChunks:
    def test_merges_two_files(self, tmp_path):
        p1 = tmp_path / "kb1.json"
        p2 = tmp_path / "kb2.json"
        _write_chunks(p1, [_chunk("a1"), _chunk("a2")])
        _write_chunks(p2, [_chunk("b1")])
        merged = merge_chunks(p1, p2)
        assert len(merged) == 3
        ids = {c["chunk_id"] for c in merged}
        assert ids == {"a1", "a2", "b1"}

    def test_merges_three_files(self, tmp_path):
        p1 = tmp_path / "kb1.json"
        p2 = tmp_path / "kb2.json"
        p3 = tmp_path / "kb3.json"
        _write_chunks(p1, [_chunk("a1")])
        _write_chunks(p2, [_chunk("b1")])
        _write_chunks(p3, [_chunk("c1")])
        merged = merge_chunks(p1, p2, p3)
        assert len(merged) == 3

    def test_preserves_order(self, tmp_path):
        p1 = tmp_path / "kb1.json"
        p2 = tmp_path / "kb2.json"
        _write_chunks(p1, [_chunk("a1"), _chunk("a2")])
        _write_chunks(p2, [_chunk("b1")])
        merged = merge_chunks(p1, p2)
        ids = [c["chunk_id"] for c in merged]
        assert ids == ["a1", "a2", "b1"]

    def test_detects_duplicate_chunk_ids(self, tmp_path):
        p1 = tmp_path / "kb1.json"
        p2 = tmp_path / "kb2.json"
        _write_chunks(p1, [_chunk("dup")])
        _write_chunks(p2, [_chunk("dup")])
        with pytest.raises(ValueError, match="Duplicate chunk_id"):
            merge_chunks(p1, p2)

    def test_empty_files(self, tmp_path):
        p1 = tmp_path / "kb1.json"
        _write_chunks(p1, [])
        merged = merge_chunks(p1)
        assert merged == []

    def test_single_file(self, tmp_path):
        p1 = tmp_path / "kb1.json"
        _write_chunks(p1, [_chunk("a1"), _chunk("a2")])
        merged = merge_chunks(p1)
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# Tests: merge_evaluation_data
# ---------------------------------------------------------------------------

class TestMergeEvaluationData:
    def test_merges_relevant_chunks_across_kbs(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        p2 = tmp_path / "eval2.json"
        _write_eval(p1, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        _write_eval(p2, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["b1"],
             "category": "direct", "expected_abstention": False},
        ])
        merged = merge_evaluation_data({"kb1": p1, "kb2": p2})
        q1 = next(q for q in merged if q["id"] == "q1")
        assert set(q1["relevant_chunks"]) == {"a1", "b1"}

    def test_question_in_only_one_kb(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        p2 = tmp_path / "eval2.json"
        _write_eval(p1, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        _write_eval(p2, [
            {"id": "q1", "question": "Q?", "relevant_chunks": [],
             "category": "direct", "expected_abstention": False},
        ])
        merged = merge_evaluation_data({"kb1": p1, "kb2": p2})
        q1 = next(q for q in merged if q["id"] == "q1")
        assert q1["relevant_chunks"] == ["a1"]

    def test_unanswerable_question_preserved(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        _write_eval(p1, [
            {"id": "q1", "question": "Q?", "relevant_chunks": [],
             "category": "unanswerable", "expected_abstention": True},
        ])
        merged = merge_evaluation_data({"kb1": p1})
        assert merged[0]["expected_abstention"] is True

    def test_deduplicates_relevant_chunks(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        p2 = tmp_path / "eval2.json"
        _write_eval(p1, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["a1", "a2"],
             "category": "direct", "expected_abstention": False},
        ])
        _write_eval(p2, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["a1", "b1"],
             "category": "direct", "expected_abstention": False},
        ])
        merged = merge_evaluation_data({"kb1": p1, "kb2": p2})
        q1 = next(q for q in merged if q["id"] == "q1")
        assert sorted(q1["relevant_chunks"]) == ["a1", "a2", "b1"]

    def test_multiple_questions(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        _write_eval(p1, [
            {"id": "q1", "question": "Q1?", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
            {"id": "q2", "question": "Q2?", "relevant_chunks": ["a2"],
             "category": "multi-source", "expected_abstention": False},
        ])
        merged = merge_evaluation_data({"kb1": p1})
        assert len(merged) == 2

    def test_three_kbs(self, tmp_path):
        p1 = tmp_path / "eval1.json"
        p2 = tmp_path / "eval2.json"
        p3 = tmp_path / "eval3.json"
        _write_eval(p1, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        _write_eval(p2, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["b1"],
             "category": "direct", "expected_abstention": False},
        ])
        _write_eval(p3, [
            {"id": "q1", "question": "Q?", "relevant_chunks": ["c1"],
             "category": "direct", "expected_abstention": False},
        ])
        merged = merge_evaluation_data({"kb1": p1, "kb2": p2, "kb3": p3})
        q1 = next(q for q in merged if q["id"] == "q1")
        assert set(q1["relevant_chunks"]) == {"a1", "b1", "c1"}


# ---------------------------------------------------------------------------
# Tests: run_merged_pool_evaluation
# ---------------------------------------------------------------------------

class TestRunMergedPoolEvaluation:
    def test_returns_report_dict(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1", "relevant text")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_merged_pool_evaluation(
            retriever=_StubRetriever(),
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["strategy"] == "merged_pool"
        assert "metrics" in report
        assert "config" in report

    def test_correct_corpus_size(self, tmp_path):
        cp1 = tmp_path / "chunks1.json"
        cp2 = tmp_path / "chunks2.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp1, [_chunk("a1"), _chunk("a2")])
        _write_chunks(cp2, [_chunk("b1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_merged_pool_evaluation(
            retriever=_StubRetriever(),
            chunk_paths={"kb1": cp1, "kb2": cp2},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["config"]["corpus_size"] == 3

    def test_combination_field(self, tmp_path):
        cp1 = tmp_path / "c1.json"
        cp2 = tmp_path / "c2.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp1, [_chunk("a1")])
        _write_chunks(cp2, [_chunk("b1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_merged_pool_evaluation(
            retriever=_StubRetriever(),
            chunk_paths={"faq-ao": cp1, "stpo": cp2},
            eval_paths={"faq-ao": ep},
            k=5,
        )
        assert report["combination"] == "faq-ao+stpo"

    def test_skips_unanswerable(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": [],
             "category": "unanswerable", "expected_abstention": True},
        ])
        report = run_merged_pool_evaluation(
            retriever=_StubRetriever(),
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["metrics"]["num_queries"] == 0

    def test_metrics_are_valid(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1", "relevant answer")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_merged_pool_evaluation(
            retriever=_StubRetriever(),
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        m = report["metrics"]
        assert m["num_queries"] == 1
        assert 0.0 <= m["precision_at_k"] <= 1.0
        assert 0.0 <= m["recall_at_k"] <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0

    def test_zero_chunk_kb_included(self, tmp_path):
        """A KB with 0 chunks (e.g., FAQ-AO) contributes nothing but does not break."""
        cp1 = tmp_path / "chunks1.json"
        cp2 = tmp_path / "chunks_empty.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp1, [_chunk("a1", "relevant answer")])
        _write_chunks(cp2, [])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_merged_pool_evaluation(
            retriever=_StubRetriever(),
            chunk_paths={"stpo": cp1, "faq-ao": cp2},
            eval_paths={"stpo": ep},
            k=5,
        )
        assert report["config"]["corpus_size"] == 1
        assert report["metrics"]["num_queries"] == 1


# ---------------------------------------------------------------------------
# Tests: run_fusion_evaluation
# ---------------------------------------------------------------------------

class TestRunFusionEvaluation:
    def test_returns_report_dict(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1", "relevant text")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["strategy"] == "fusion"
        assert "metrics" in report
        assert "config" in report
        assert "k_rrf" in report["config"]

    def test_correct_corpus_size(self, tmp_path):
        cp1 = tmp_path / "c1.json"
        cp2 = tmp_path / "c2.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp1, [_chunk("a1"), _chunk("a2")])
        _write_chunks(cp2, [_chunk("b1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"kb1": cp1, "kb2": cp2},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["config"]["corpus_size"] == 3

    def test_combination_field(self, tmp_path):
        cp1 = tmp_path / "c1.json"
        cp2 = tmp_path / "c2.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp1, [_chunk("a1")])
        _write_chunks(cp2, [_chunk("b1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"faq-ao": cp1, "stpo": cp2},
            eval_paths={"faq-ao": ep},
            k=5,
        )
        assert report["combination"] == "faq-ao+stpo"

    def test_skips_unanswerable(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": [],
             "category": "unanswerable", "expected_abstention": True},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["metrics"]["num_queries"] == 0

    def test_metrics_are_valid(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1", "relevant answer")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        m = report["metrics"]
        assert m["num_queries"] == 1
        assert 0.0 <= m["precision_at_k"] <= 1.0
        assert 0.0 <= m["recall_at_k"] <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0

    def test_retriever_type_in_config(self, tmp_path):
        cp = tmp_path / "chunks.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp, [_chunk("a1")])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"kb1": cp},
            eval_paths={"kb1": ep},
            k=5,
        )
        assert report["config"]["retriever_type"] == "_StubRetriever"

    def test_zero_chunk_kb_included(self, tmp_path):
        """A KB with 0 chunks (e.g., FAQ-AO) contributes nothing but does not break."""
        cp1 = tmp_path / "c1.json"
        cp2 = tmp_path / "c_empty.json"
        ep = tmp_path / "eval.json"
        _write_chunks(cp1, [_chunk("a1", "relevant answer")])
        _write_chunks(cp2, [])
        _write_eval(ep, [
            {"id": "q1", "question": "test", "relevant_chunks": ["a1"],
             "category": "direct", "expected_abstention": False},
        ])
        report = run_fusion_evaluation(
            retriever_factory=_StubRetriever,
            chunk_paths={"stpo": cp1, "faq-ao": cp2},
            eval_paths={"stpo": ep},
            k=5,
        )
        assert report["config"]["corpus_size"] == 1
        assert report["metrics"]["num_queries"] == 1
