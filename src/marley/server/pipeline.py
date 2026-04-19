"""Abstention-aware pipeline orchestrator.

Combines retrieval, score normalization, threshold filtering (Level 1),
generation, and LLM abstention detection (Level 2) into a single function
that returns an AbstentionResult.

This module lives in ``server/`` because it orchestrates across multiple
pipeline stages (retrieval, generation, abstention detection).
"""

from __future__ import annotations

from typing import Any

from src.marley.abstention.detection import detect_abstention, extract_abstention_reason
from src.marley.models.abstention import AbstentionResult
from src.marley.models.constants import DEFAULT_K, DEFAULT_THRESHOLD
from src.marley.models.generation import Generator
from src.marley.models.retrieval import Retriever
from src.marley.models.scoring import (
    compute_confidence,
    compute_fusion_confidence,
    filter_by_threshold,
    normalize_scores,
)
from src.marley.retrieval.fusion import FusionRetriever

_LEVEL1_REASON = "retrieval confidence below threshold"


def _results_to_dicts(results: list) -> list[dict[str, Any]]:
    """Serialize RetrievalResult objects to plain dicts."""
    return [
        {"chunk_id": r.chunk_id, "text": r.text, "score": r.score, "metadata": r.metadata}
        for r in results
    ]


def run_with_abstention(
    query: str,
    retriever: Retriever,
    generator: Generator,
    *,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    normalization_strategy: str = "vector",
    normalization_params: dict[str, Any] | None = None,
    fusion_sub_strategy: str | None = None,
) -> AbstentionResult:
    """Run the full abstention-aware pipeline.

    Steps:
        1. Retrieve top-k chunks.
        2. Normalize scores using the specified strategy.
        3. Compute confidence — top-1 normalized score for ordinary
           retrievers, or the fusion-aware aggregate for a
           :class:`FusionRetriever` (see :func:`compute_fusion_confidence`).
        4. Level-1 abstention check:
           - fusion: abstain iff ``confidence < threshold`` (no filter;
             RRF scores of disjoint-KB sub-retrievers do not discriminate
             per query, so threshold filtering on the fused output is
             degenerate by construction);
           - non-fusion: filter chunks by threshold and abstain iff the
             filtered set is empty.
        5. Generate answer from the retained context.
        6. Detect abstention in LLM output (Level 2).
        7. Return AbstentionResult.

    Args:
        query: The user question.
        retriever: Retriever instance (must be indexed).
        generator: Generator instance.
        k: Number of chunks to retrieve.
        threshold: Minimum normalized score to keep a chunk (non-fusion)
            or minimum fusion confidence to proceed (fusion).
        normalization_strategy: One of 'bm25', 'vector', 'rrf'. For a
            fusion retriever this governs how the fused output is
            normalised for downstream consumers.
        normalization_params: Extra kwargs for normalize_scores
            (e.g., bm25_k, rrf_n_retrievers, rrf_k).
        fusion_sub_strategy: Normalisation strategy of the sub-retrievers
            when ``retriever`` is a :class:`FusionRetriever`. Required
            to enable fusion-aware confidence; when omitted or the
            retriever is not a FusionRetriever, confidence falls back to
            the top-1 normalised score of the fused output.

    Returns:
        AbstentionResult with the pipeline outcome.
    """
    norm_params = normalization_params or {}
    is_fusion = isinstance(retriever, FusionRetriever)

    # Step 1: Retrieve
    raw_results = retriever.retrieve(query, k=k)

    # Step 2: Normalize
    normalized = normalize_scores(raw_results, normalization_strategy, **norm_params)

    # Step 3: Confidence (fusion-aware when applicable)
    if is_fusion and fusion_sub_strategy is not None:
        confidence = compute_fusion_confidence(
            retriever.last_sub_results,  # type: ignore[attr-defined]
            fusion_sub_strategy,
            **({"bm25_k": norm_params["bm25_k"]} if "bm25_k" in norm_params else {}),
        )
    else:
        confidence = compute_confidence(normalized)

    # Step 4: Level-1 check
    if is_fusion and fusion_sub_strategy is not None:
        # Fusion path: threshold on the fusion-aware confidence; retain
        # the full fused output so Level-2 and the UI can still cite the
        # top chunks regardless of individual RRF scores.
        if confidence < threshold:
            return AbstentionResult(
                abstained=True,
                level=1,
                reason=_LEVEL1_REASON,
                answer="",
                confidence=confidence,
                retrieval_results=_results_to_dicts(normalized),
                model="",
            )
        filtered = normalized
    else:
        filtered = filter_by_threshold(normalized, threshold)
        if not filtered:
            return AbstentionResult(
                abstained=True,
                level=1,
                reason=_LEVEL1_REASON,
                answer="",
                confidence=confidence,
                retrieval_results=_results_to_dicts(normalized),
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
            retrieval_results=_results_to_dicts(filtered),
            model=gen_result.model,
        )

    # Step 8: Normal answer
    return AbstentionResult(
        abstained=False,
        level=None,
        reason="",
        answer=gen_result.answer,
        confidence=confidence,
        retrieval_results=_results_to_dicts(filtered),
        model=gen_result.model,
    )
