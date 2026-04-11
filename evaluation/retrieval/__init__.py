"""Retrieval evaluation subpackage."""

from evaluation.retrieval.combined import (
    run_fusion_evaluation,
    run_merged_pool_evaluation,
)
from evaluation.retrieval.evaluate import run_and_report, run_evaluation
from evaluation.retrieval.rrf_tuning import sweep_fusion_k_rrf, sweep_hybrid_k_rrf
from evaluation.retrieval.metrics import (
    RetrievalMetrics,
    evaluate_retriever,
    f1_at_k,
    jaccard_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "RetrievalMetrics",
    "evaluate_retriever",
    "f1_at_k",
    "jaccard_at_k",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "run_and_report",
    "run_evaluation",
    "run_fusion_evaluation",
    "run_merged_pool_evaluation",
    "sweep_fusion_k_rrf",
    "sweep_hybrid_k_rrf",
]
