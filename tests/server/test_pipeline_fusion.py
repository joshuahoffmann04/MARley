"""Regression tests for the fusion-aware branch of run_with_abstention.

These tests lock in that when the retriever is a FusionRetriever and a
``fusion_sub_strategy`` is provided, confidence comes from
``compute_fusion_confidence`` over the raw sub-retriever outputs — not
from the top-1 score of the fused output, which on disjoint KBs is a
per-query constant.
"""

from __future__ import annotations

from src.marley.models.retrieval import RetrievalResult, rrf_fuse
from src.marley.retrieval.fusion import FusionRetriever
from src.marley.server.pipeline import run_with_abstention
from tests.conftest import FixedRetriever, StubGenerator


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, text=f"t-{chunk_id}", score=score, metadata={})


def _fusion(
    results_per_sub: list[list[RetrievalResult]],
    k_rrf: int = 1,
) -> FusionRetriever:
    subs = [FixedRetriever(rs) for rs in results_per_sub]
    return FusionRetriever(subs, k_rrf=k_rrf)


class TestRunWithAbstentionFusion:
    def test_fusion_confidence_is_used_when_sub_strategy_provided(self) -> None:
        # Two sub-retrievers, one with a strong vector hit, one weak.
        retriever = _fusion([
            [_result("A-1", 0.9), _result("A-2", 0.2)],
            [_result("B-1", 0.1)],
        ])
        generator = StubGenerator(answer="ok")
        result = run_with_abstention(
            "query",
            retriever,
            generator,
            k=3,
            threshold=0.3,
            normalization_strategy="rrf",
            normalization_params={"rrf_n_retrievers": 2, "rrf_k": 1},
            fusion_sub_strategy="vector",
        )
        # Fusion-aware confidence = max over sub-normalised top-1
        # scores. With "vector" strategy the identity normalisation keeps
        # the 0.9 top-1 from the first sub-retriever.
        assert result.confidence == 0.9
        assert result.abstained is False

    def test_fusion_abstains_when_confidence_below_threshold(self) -> None:
        # Every sub-retriever is weak; fusion confidence must stay low.
        retriever = _fusion([
            [_result("A-1", 0.1)],
            [_result("B-1", 0.05)],
        ])
        generator = StubGenerator(answer="should not be called")
        result = run_with_abstention(
            "query",
            retriever,
            generator,
            k=3,
            threshold=0.5,
            normalization_strategy="rrf",
            normalization_params={"rrf_n_retrievers": 2, "rrf_k": 1},
            fusion_sub_strategy="vector",
        )
        assert result.abstained is True
        assert result.level == 1
        assert result.confidence == 0.1  # max over sub-confidences

    def test_fusion_without_sub_strategy_falls_back_to_top1(self) -> None:
        """Omitting fusion_sub_strategy keeps legacy top-1 behaviour."""
        retriever = _fusion([
            [_result("A-1", 0.8)],
            [_result("B-1", 0.4)],
        ])
        generator = StubGenerator(answer="ok")
        result = run_with_abstention(
            "query",
            retriever,
            generator,
            k=3,
            threshold=0.1,
            normalization_strategy="rrf",
            normalization_params={"rrf_n_retrievers": 2, "rrf_k": 1},
        )
        # Fall-back path must still answer; confidence is the fused
        # top-1 normalised score (rrf normalisation), bounded in [0, 1].
        assert result.abstained is False
        assert 0.0 < result.confidence <= 1.0

    def test_fusion_confidence_differs_from_top1_on_disjoint_corpora(
        self,
    ) -> None:
        """Regression guard for F-1-14: two queries with very different
        sub-retriever strengths must produce different fusion
        confidences, even though their *fused* top-1 RRF scores are the
        same on disjoint-KB setups."""
        strong = _fusion([
            [_result("A-1", 0.95)],
            [_result("B-1", 0.10)],
        ])
        weak = _fusion([
            [_result("A-1", 0.20)],
            [_result("B-1", 0.10)],
        ])
        gen = StubGenerator(answer="ok")
        common_kwargs = {
            "k": 3,
            "threshold": 0.0,
            "normalization_strategy": "rrf",
            "normalization_params": {"rrf_n_retrievers": 2, "rrf_k": 1},
            "fusion_sub_strategy": "vector",
        }
        r_strong = run_with_abstention("q", strong, gen, **common_kwargs)
        r_weak = run_with_abstention("q", weak, gen, **common_kwargs)
        # Must discriminate — the whole point of the fix.
        assert r_strong.confidence > r_weak.confidence


class TestFusionLastSubResultsAfterPipeline:
    def test_last_sub_results_populated_after_pipeline_run(self) -> None:
        retriever = _fusion([
            [_result("A-1", 0.5), _result("A-2", 0.3)],
            [_result("B-1", 0.4)],
        ])
        gen = StubGenerator(answer="ok")
        run_with_abstention(
            "q",
            retriever,
            gen,
            k=3,
            threshold=0.0,
            normalization_strategy="rrf",
            normalization_params={"rrf_n_retrievers": 2, "rrf_k": 1},
            fusion_sub_strategy="vector",
        )
        assert len(retriever.last_sub_results) == 2
        assert retriever.last_sub_results[0][0].chunk_id == "A-1"
        assert retriever.last_sub_results[1][0].chunk_id == "B-1"


class TestRrfFuseSanity:
    """Tiny unit check that rrf_fuse still produces a non-empty fused
    output from the FixedRetriever-based sub-results used above."""

    def test_rrf_fuse_merges_disjoint_subresults(self) -> None:
        fused = rrf_fuse(
            [[_result("A-1", 0.9)], [_result("B-1", 0.5)]],
            k_rrf=1,
            k=3,
        )
        assert {r.chunk_id for r in fused} == {"A-1", "B-1"}
