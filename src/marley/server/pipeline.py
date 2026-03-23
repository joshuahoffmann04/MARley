"""Abstention-aware pipeline orchestrator.

Combines retrieval, score normalization, threshold filtering (Level 1),
generation, and LLM abstention detection (Level 2) into a single function
that returns an AbstentionResult.

This module lives in ``server/`` because it orchestrates across multiple
pipeline stages (retrieval, generation, abstention detection).
"""

from __future__ import annotations

from src.marley.abstention.detection import detect_abstention, extract_abstention_reason
from src.marley.models.abstention import AbstentionResult
from src.marley.models.constants import DEFAULT_K, DEFAULT_THRESHOLD
from src.marley.models.generation import Generator
from src.marley.models.retrieval import Retriever
from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)

_LEVEL1_REASON = "retrieval confidence below threshold"


def run_with_abstention(
    query: str,
    retriever: Retriever,
    generator: Generator,
    *,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    normalization_strategy: str = "vector",
    normalization_params: dict | None = None,
) -> AbstentionResult:
    """Run the full abstention-aware pipeline.

    Steps:
        1. Retrieve top-k chunks.
        2. Normalize scores using the specified strategy.
        3. Compute confidence (top-1 normalized score).
        4. Filter chunks below threshold.
        5. If no chunks remain: abstain (Level 1).
        6. Generate answer from filtered context.
        7. Detect abstention in LLM output (Level 2).
        8. Return AbstentionResult.

    Args:
        query: The user question.
        retriever: Retriever instance (must be indexed).
        generator: Generator instance.
        k: Number of chunks to retrieve.
        threshold: Minimum normalized score to keep a chunk.
        normalization_strategy: One of 'bm25', 'vector', 'rrf'.
        normalization_params: Extra kwargs for normalize_scores
            (e.g., bm25_k, rrf_n_retrievers, rrf_k).

    Returns:
        AbstentionResult with the pipeline outcome.
    """
    norm_params = normalization_params or {}

    # Step 1: Retrieve
    raw_results = retriever.retrieve(query, k=k)

    # Step 2: Normalize
    normalized = normalize_scores(raw_results, normalization_strategy, **norm_params)

    # Step 3: Confidence
    confidence = compute_confidence(normalized)

    # Step 4: Filter
    filtered = filter_by_threshold(normalized, threshold)

    # Step 5: Level 1 check
    if not filtered:
        return AbstentionResult(
            abstained=True,
            level=1,
            reason=_LEVEL1_REASON,
            answer="",
            confidence=confidence,
            retrieval_results=[
                {"chunk_id": r.chunk_id, "text": r.text, "score": r.score}
                for r in normalized
            ],
            model="",
        )

    # Step 6: Generate
    context = [
        {"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata}
        for r in filtered
    ]
    gen_result = generator.generate(query, context)

    # Step 7: Level 2 check
    if detect_abstention(gen_result.answer):
        return AbstentionResult(
            abstained=True,
            level=2,
            reason=extract_abstention_reason(gen_result.answer),
            answer="",
            confidence=confidence,
            retrieval_results=[
                {"chunk_id": r.chunk_id, "text": r.text, "score": r.score}
                for r in filtered
            ],
            model=gen_result.model,
        )

    # Step 8: Normal answer
    return AbstentionResult(
        abstained=False,
        level=None,
        reason="",
        answer=gen_result.answer,
        confidence=confidence,
        retrieval_results=[
            {"chunk_id": r.chunk_id, "text": r.text, "score": r.score}
            for r in filtered
        ],
        model=gen_result.model,
    )
