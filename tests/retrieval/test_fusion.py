"""Tests for the RRF fusion utility function."""

import pytest

from src.marley.retrieval import RetrievalResult, rrf_fuse


def _r(chunk_id: str, score: float = 1.0, text: str = "") -> RetrievalResult:
    """Shorthand for creating a RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text or f"text-{chunk_id}",
        score=score,
        metadata={"source": chunk_id},
    )


class TestRRFFuse:
    """Tests for the standalone rrf_fuse() function."""

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_input_returns_empty(self):
        assert rrf_fuse([], k=5) == []

    def test_single_empty_list_returns_empty(self):
        assert rrf_fuse([[]], k=5) == []

    def test_multiple_empty_lists_returns_empty(self):
        assert rrf_fuse([[], [], []], k=5) == []

    def test_mixed_empty_and_nonempty_lists(self):
        results = [_r("c1", 1.0), _r("c2", 0.5)]
        fused = rrf_fuse([[], results, []], k=5)
        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"

    # ------------------------------------------------------------------
    # Single list
    # ------------------------------------------------------------------

    def test_single_list_preserves_order(self):
        results = [_r("c1", 1.0), _r("c2", 0.5), _r("c3", 0.2)]
        fused = rrf_fuse([results], k=3)
        assert [r.chunk_id for r in fused] == ["c1", "c2", "c3"]

    def test_single_list_respects_k(self):
        results = [_r("c1", 1.0), _r("c2", 0.5), _r("c3", 0.2)]
        fused = rrf_fuse([results], k=1)
        assert len(fused) == 1
        assert fused[0].chunk_id == "c1"

    # ------------------------------------------------------------------
    # Two lists (backward-compatible with HybridRetriever)
    # ------------------------------------------------------------------

    def test_two_lists_shared_document_ranks_higher(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c2", 0.9), _r("c3", 0.4)]
        fused = rrf_fuse([list_a, list_b], k=3)
        # c2 appears in both -> highest RRF score
        assert fused[0].chunk_id == "c2"
        assert set(r.chunk_id for r in fused) == {"c1", "c2", "c3"}

    def test_two_lists_no_overlap(self):
        list_a = [_r("c1", 1.0)]
        list_b = [_r("c2", 0.9)]
        fused = rrf_fuse([list_a, list_b], k=2)
        assert len(fused) == 2
        ids = {r.chunk_id for r in fused}
        assert ids == {"c1", "c2"}

    def test_two_lists_full_overlap(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c1", 0.9), _r("c2", 0.4)]
        fused = rrf_fuse([list_a, list_b], k=2)
        assert len(fused) == 2
        # No duplicates
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))

    # ------------------------------------------------------------------
    # Three or more lists (new capability)
    # ------------------------------------------------------------------

    def test_three_lists_fuses_correctly(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c2", 0.9), _r("c3", 0.4)]
        list_c = [_r("c2", 0.8), _r("c4", 0.3)]
        fused = rrf_fuse([list_a, list_b, list_c], k=4)
        # c2 appears in all three -> highest RRF score
        assert fused[0].chunk_id == "c2"
        assert set(r.chunk_id for r in fused) == {"c1", "c2", "c3", "c4"}

    def test_three_lists_document_in_all_beats_document_in_two(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c1", 0.9), _r("c2", 0.4)]
        list_c = [_r("c1", 0.8)]
        fused = rrf_fuse([list_a, list_b, list_c], k=2)
        # c1 appears in all three, c2 only in two
        assert fused[0].chunk_id == "c1"
        assert fused[1].chunk_id == "c2"

    # ------------------------------------------------------------------
    # RRF score computation
    # ------------------------------------------------------------------

    def test_rrf_score_formula_two_lists(self):
        list_a = [_r("c1", 1.0)]
        list_b = [_r("c1", 0.9)]
        fused = rrf_fuse([list_a, list_b], k_rrf=60, k=1)
        # c1 at rank 1 in both: score = 2 * 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(fused[0].score - expected) < 1e-10

    def test_rrf_score_formula_different_ranks(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c2", 0.9), _r("c1", 0.4)]
        fused = rrf_fuse([list_a, list_b], k_rrf=60, k=2)
        # c1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61  (same!)
        expected = 1.0 / 61.0 + 1.0 / 62.0
        for r in fused:
            assert abs(r.score - expected) < 1e-10

    def test_custom_k_rrf_affects_scores(self):
        results = [_r("c1", 1.0)]
        fused_small = rrf_fuse([results], k_rrf=1, k=1)
        fused_large = rrf_fuse([results], k_rrf=60, k=1)
        # k_rrf=1: score = 1/(1+1) = 0.5
        # k_rrf=60: score = 1/(60+1) ≈ 0.016
        assert fused_small[0].score > fused_large[0].score

    # ------------------------------------------------------------------
    # Metadata handling
    # ------------------------------------------------------------------

    def test_metadata_from_highest_scoring_source(self):
        r1 = RetrievalResult(chunk_id="c1", text="low", score=0.3, metadata={"src": "a"})
        r2 = RetrievalResult(chunk_id="c1", text="high", score=0.9, metadata={"src": "b"})
        fused = rrf_fuse([[r1], [r2]], k=1)
        assert fused[0].text == "high"
        assert fused[0].metadata == {"src": "b"}

    def test_metadata_preserved_for_unique_documents(self):
        r1 = RetrievalResult(chunk_id="c1", text="t1", score=1.0, metadata={"k": "v1"})
        r2 = RetrievalResult(chunk_id="c2", text="t2", score=0.9, metadata={"k": "v2"})
        fused = rrf_fuse([[r1], [r2]], k=2)
        meta_map = {r.chunk_id: r.metadata for r in fused}
        assert meta_map["c1"] == {"k": "v1"}
        assert meta_map["c2"] == {"k": "v2"}

    # ------------------------------------------------------------------
    # Output properties
    # ------------------------------------------------------------------

    def test_scores_sorted_descending(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5), _r("c3", 0.2)]
        list_b = [_r("c3", 0.9), _r("c1", 0.4), _r("c2", 0.1)]
        fused = rrf_fuse([list_a, list_b], k=3)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_positive(self):
        list_a = [_r("c1", 1.0)]
        list_b = [_r("c2", 0.5)]
        fused = rrf_fuse([list_a, list_b], k=2)
        assert all(r.score > 0 for r in fused)

    def test_no_duplicate_chunk_ids(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c1", 0.9), _r("c2", 0.4)]
        list_c = [_r("c1", 0.8), _r("c2", 0.3)]
        fused = rrf_fuse([list_a, list_b, list_c], k=5)
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))

    def test_returns_retrieval_result_type(self):
        fused = rrf_fuse([[_r("c1", 1.0)]], k=1)
        assert isinstance(fused[0], RetrievalResult)

    def test_k_limits_output(self):
        results = [_r(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        fused = rrf_fuse([results], k=3)
        assert len(fused) == 3

    # ------------------------------------------------------------------
    # Weighted RRF
    # ------------------------------------------------------------------

    def test_uniform_weights_match_no_weights(self):
        list_a = [_r("c1", 1.0), _r("c2", 0.5)]
        list_b = [_r("c2", 0.9), _r("c3", 0.4)]
        fused_none = rrf_fuse([list_a, list_b], k_rrf=60, k=3)
        fused_uniform = rrf_fuse([list_a, list_b], k_rrf=60, k=3, weights=[1.0, 1.0])
        for r_none, r_uniform in zip(fused_none, fused_uniform):
            assert r_none.chunk_id == r_uniform.chunk_id
            assert abs(r_none.score - r_uniform.score) < 1e-10

    def test_double_weight_boosts_retriever(self):
        list_a = [_r("c1", 1.0)]  # only in list_a
        list_b = [_r("c2", 0.9)]  # only in list_b
        # Equal weights: both at rank 1, same RRF score
        fused_equal = rrf_fuse([list_a, list_b], k_rrf=60, k=2, weights=[1.0, 1.0])
        assert fused_equal[0].score == fused_equal[1].score
        # Double weight on list_b: c2 should rank higher
        fused_boost = rrf_fuse([list_a, list_b], k_rrf=60, k=2, weights=[1.0, 2.0])
        assert fused_boost[0].chunk_id == "c2"
        assert fused_boost[0].score > fused_boost[1].score

    def test_weights_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected 2 weights"):
            rrf_fuse([[_r("c1")], [_r("c2")]], weights=[1.0])

    def test_weights_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            rrf_fuse([[_r("c1")], [_r("c2")]], weights=[1.0, -0.5])

    def test_weights_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            rrf_fuse([[_r("c1")], [_r("c2")]], weights=[1.0, 0.0])

    def test_weighted_score_formula(self):
        list_a = [_r("c1", 1.0)]
        list_b = [_r("c1", 0.9)]
        fused = rrf_fuse([list_a, list_b], k_rrf=60, k=1, weights=[1.0, 3.0])
        # c1 at rank 1 in both: 1.0/(60+1) + 3.0/(60+1) = 4.0/61
        expected = 4.0 / 61.0
        assert abs(fused[0].score - expected) < 1e-10


# ======================================================================
# FusionRetriever tests
# ======================================================================


class TestFusionRetriever:
    """Tests for the FusionRetriever wrapper class."""

    # ------------------------------------------------------------------
    # Setup helper
    # ------------------------------------------------------------------

    class _StubRetriever:
        """Minimal retriever returning pre-set results."""

        def __init__(self, results: list[RetrievalResult]) -> None:
            self._results = results
            self._size = len(results)

        def index(self, corpus: list[dict]) -> None:
            self._size = len(corpus)

        def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
            return self._results[:k]

        @property
        def size(self) -> int:
            return self._size

    # ------------------------------------------------------------------
    # Core behavior
    # ------------------------------------------------------------------

    def test_single_sub_retriever_pass_through(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r("c1", 1.0), _r("c2", 0.5)])
        fusion = FusionRetriever([stub])
        results = fusion.retrieve("query", k=5)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"

    def test_two_sub_retrievers_fuse_results(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub_a = self._StubRetriever([_r("c1", 1.0), _r("c2", 0.5)])
        stub_b = self._StubRetriever([_r("c2", 0.9), _r("c3", 0.4)])
        fusion = FusionRetriever([stub_a, stub_b])
        results = fusion.retrieve("query", k=3)
        # c2 appears in both -> highest RRF score
        assert results[0].chunk_id == "c2"
        assert len(results) == 3

    def test_k_limits_output(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r(f"c{i}", 1.0 - i * 0.1) for i in range(10)])
        fusion = FusionRetriever([stub])
        results = fusion.retrieve("query", k=3)
        assert len(results) == 3

    def test_size_sums_sub_retrievers(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub_a = self._StubRetriever([_r("c1", 1.0)])
        stub_a._size = 10
        stub_b = self._StubRetriever([_r("c2", 0.9)])
        stub_b._size = 20
        fusion = FusionRetriever([stub_a, stub_b])
        assert fusion.size == 30

    def test_index_raises_not_implemented_error(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r("c1", 1.0)])
        fusion = FusionRetriever([stub])
        with pytest.raises(NotImplementedError, match="pre-indexed"):
            fusion.index([])

    def test_empty_retrievers_raises_value_error(self):
        from src.marley.retrieval.fusion import FusionRetriever

        with pytest.raises(ValueError, match="at least one"):
            FusionRetriever([])

    def test_scores_are_rrf_scores(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub_a = self._StubRetriever([_r("c1", 1.0)])
        stub_b = self._StubRetriever([_r("c1", 0.9)])
        fusion = FusionRetriever([stub_a, stub_b], k_rrf=60)
        results = fusion.retrieve("query", k=1)
        # c1 at rank 1 in both: score = 2 * 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(results[0].score - expected) < 1e-10

    def test_custom_k_rrf_applied(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r("c1", 1.0)])
        fusion_small = FusionRetriever([stub], k_rrf=1)
        fusion_large = FusionRetriever([stub], k_rrf=60)
        r_small = fusion_small.retrieve("query", k=1)
        r_large = fusion_large.retrieve("query", k=1)
        # k_rrf=1: score = 1/(1+1) = 0.5, k_rrf=60: score = 1/(60+1)
        assert r_small[0].score > r_large[0].score

    def test_weights_passed_through(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub_a = self._StubRetriever([_r("c1", 1.0)])
        stub_b = self._StubRetriever([_r("c2", 0.9)])
        fusion = FusionRetriever([stub_a, stub_b], weights=[1.0, 2.0])
        results = fusion.retrieve("query", k=2)
        # c2 gets double weight -> should rank first
        assert results[0].chunk_id == "c2"

    def test_weights_wrong_length_raises(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r("c1", 1.0)])
        with pytest.raises(ValueError, match="Expected 1 weights"):
            FusionRetriever([stub], weights=[1.0, 2.0])


class TestFusionRetrieverSubResultsCache:
    """FusionRetriever caches raw sub-retriever outputs so that
    downstream code can compute a Fusion-aware confidence without re-running
    the sub-queries."""

    class _StubRetriever:
        def __init__(self, results):
            self._results = results
            self._size = len(results)

        def index(self, corpus):
            self._size = len(corpus)

        def retrieve(self, query, k=5):
            return self._results[:k]

        @property
        def size(self):
            return self._size

    def test_last_sub_results_empty_before_retrieve(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r("c1", 1.0)])
        fusion = FusionRetriever([stub])
        assert fusion.last_sub_results == []

    def test_last_sub_results_populated_after_retrieve(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub_a = self._StubRetriever([_r("a1", 0.9)])
        stub_b = self._StubRetriever([_r("b1", 0.4)])
        fusion = FusionRetriever([stub_a, stub_b])
        fusion.retrieve("query", k=1)

        subs = fusion.last_sub_results
        assert len(subs) == 2
        assert subs[0][0].chunk_id == "a1"
        assert subs[1][0].chunk_id == "b1"

    def test_last_sub_results_reflects_latest_query(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub_a = self._StubRetriever([_r("a1", 0.9), _r("a2", 0.2)])
        fusion = FusionRetriever([stub_a])
        fusion.retrieve("q1", k=2)
        first = fusion.last_sub_results
        fusion.retrieve("q2", k=1)
        second = fusion.last_sub_results
        assert len(first[0]) == 2
        assert len(second[0]) == 1

    def test_sub_retrievers_property_is_read_only_view(self):
        from src.marley.retrieval.fusion import FusionRetriever

        stub = self._StubRetriever([_r("c1", 1.0)])
        fusion = FusionRetriever([stub])
        view = fusion.sub_retrievers
        assert len(view) == 1
        # Mutating the returned list must not affect the retriever.
        view.clear()
        assert len(fusion.sub_retrievers) == 1
