"""Tests for the evaluation metrics module."""

import pytest

from evaluation.retrieval.metrics import (
    RetrievalMetrics,
    evaluate_retriever,
    f1_at_k,
    jaccard_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)

    def test_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_k_limits_results(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 1) == 1.0


class TestRecallAtK:
    def test_all_recalled(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_none_recalled(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_k_limits_results(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 1) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self):
        assert mrr(["x", "a", "b"], {"a"}) == 0.5

    def test_third_is_relevant(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_none_relevant(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0


class TestF1AtK:
    def test_perfect(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # P@2=1.0, R@2=1.0 → F1=1.0
        assert f1_at_k(retrieved, relevant, 2) == 1.0

    def test_zero(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert f1_at_k(retrieved, relevant, 2) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        # P@3=1/3, R@3=1/2, F1 = 2*(1/3)*(1/2) / (1/3 + 1/2) = (1/3)/(5/6) = 2/5
        assert f1_at_k(retrieved, relevant, 3) == pytest.approx(2 / 5)

    def test_empty_relevant(self):
        # P@2=0, R@2=0 → F1=0
        assert f1_at_k(["a", "b"], set(), 2) == 0.0

    def test_k_zero(self):
        assert f1_at_k(["a"], {"a"}, 0) == 0.0

    def test_high_precision_low_recall(self):
        retrieved = ["a"]
        relevant = {"a", "b", "c", "d"}
        # P@1=1.0, R@1=0.25, F1 = 2*1.0*0.25 / (1.0+0.25) = 0.5/1.25 = 0.4
        assert f1_at_k(retrieved, relevant, 1) == pytest.approx(0.4)


class TestJaccardAtK:
    def test_perfect_overlap(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert jaccard_at_k(retrieved, relevant, 2) == 1.0

    def test_no_overlap(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        # intersection=0, union=4
        assert jaccard_at_k(retrieved, relevant, 2) == 0.0

    def test_partial_overlap(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        # intersection={a}=1, union={a,b,x,y}=4 → 1/4
        assert jaccard_at_k(retrieved, relevant, 3) == pytest.approx(1 / 4)

    def test_retrieved_subset_of_relevant(self):
        retrieved = ["a"]
        relevant = {"a", "b", "c"}
        # intersection={a}=1, union={a,b,c}=3 → 1/3
        assert jaccard_at_k(retrieved, relevant, 1) == pytest.approx(1 / 3)

    def test_empty_both(self):
        assert jaccard_at_k([], set(), 0) == 0.0

    def test_k_limits_retrieved(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        # k=2: top_k={a,b}, intersection={a,b}=2, union={a,b}=2 → 1.0
        assert jaccard_at_k(retrieved, relevant, 2) == 1.0


class TestEvaluateRetriever:
    def test_perfect_retrieval(self):
        results = [
            (["a", "b"], {"a", "b"}),
            (["c", "d"], {"c", "d"}),
        ]
        metrics = evaluate_retriever(results, k=2)
        assert metrics.precision_at_k == 1.0
        assert metrics.recall_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.f1_at_k == 1.0
        assert metrics.jaccard_at_k == 1.0
        assert metrics.k == 2
        assert metrics.num_queries == 2

    def test_no_results(self):
        metrics = evaluate_retriever([], k=5)
        assert metrics.precision_at_k == 0.0
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.f1_at_k == 0.0
        assert metrics.jaccard_at_k == 0.0
        assert metrics.num_queries == 0

    def test_mixed_results(self):
        results = [
            (["a", "x"], {"a"}),      # P@2=0.5, R@2=1.0, RR=1.0, F1=2/3, J=1/2
            (["x", "y"], {"a"}),      # P@2=0.0, R@2=0.0, RR=0.0, F1=0.0, J=0/3=0.0
        ]
        metrics = evaluate_retriever(results, k=2)
        assert metrics.precision_at_k == pytest.approx(0.25)
        assert metrics.recall_at_k == pytest.approx(0.5)
        assert metrics.mrr == pytest.approx(0.5)
        assert metrics.f1_at_k == pytest.approx(1 / 3)
        assert metrics.jaccard_at_k == pytest.approx(0.25)

    def test_returns_dataclass(self):
        metrics = evaluate_retriever([], k=3)
        assert isinstance(metrics, RetrievalMetrics)
