"""Retrieval data classes, abstract base, and shared algorithms.

Defines the shared RetrievalResult dataclass, the Retriever abstract
base class, corpus loading/validation functions, and the Reciprocal
Rank Fusion (RRF) algorithm.  All retriever implementations import
from this module to ensure a single source of truth for the retrieval
interface, its input contract, and the shared fusion algorithm.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.marley.models.constants import DEFAULT_K, DEFAULT_K_RRF


@dataclass
class RetrievalResult:
    """A single retrieval hit with its score."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class Retriever(ABC):
    """Abstract base for all retrieval strategies.

    Subclasses must implement ``index``, ``retrieve``, and the ``size``
    property.  The corpus passed to ``index`` is a list of dicts, each
    containing at least ``chunk_id``, ``text``, and ``metadata`` keys.
    """

    @abstractmethod
    def index(self, corpus: list[dict[str, Any]]) -> None:
        """Build the retrieval index from a list of chunk dicts."""

    @abstractmethod
    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a query.

        Returns results sorted by descending relevance score.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed chunks."""


# ---------------------------------------------------------------------------
# Corpus loading and validation
# ---------------------------------------------------------------------------


def load_chunks(chunk_path: str | Path) -> list[dict[str, Any]]:
    """Load the ``chunks`` list from a JSON file produced by the chunking pipeline.

    The JSON file must contain a top-level ``chunks`` key whose value is
    a list of dicts with ``chunk_id``, ``text``, and ``metadata`` keys.
    """
    path = Path(chunk_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["chunks"]


def validate_corpus(corpus: list[dict[str, Any]]) -> None:
    """Validate that every dict in the corpus has the required keys.

    Raises ``ValueError`` with a descriptive message if any dict is
    missing ``chunk_id``, ``text``, or ``metadata``.
    """
    required = {"chunk_id", "text", "metadata"}
    for i, doc in enumerate(corpus):
        missing = required - doc.keys()
        if missing:
            msg = (
                f"Corpus item at index {i} is missing required "
                f"keys: {', '.join(sorted(missing))}."
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------


def rrf_fuse(
    result_lists: list[list[RetrievalResult]],
    k_rrf: int = DEFAULT_K_RRF,
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Accepts any number of result lists sorted by descending relevance.
    The fused score for each document is computed as:

        score(d) = sum( weight_i / (k_rrf + rank_i(d)) )

    where rank_i(d) is the 1-based rank of document d in list i, and
    weight_i defaults to 1.0 for all lists (uniform weighting).  Custom
    weights allow boosting or dampening individual retrievers or
    knowledge bases.

    When a document appears in multiple lists, its text and metadata
    are taken from the list where it had the highest original score.

    Reference:
        Cormack, Clarke & Buettcher (2009).  Reciprocal rank fusion
        outperforms condorcet and individual rank learning methods.
    """
    if not result_lists:
        return []

    if weights is not None:
        if len(weights) != len(result_lists):
            msg = (
                f"Expected {len(result_lists)} weights "
                f"(one per result list), got {len(weights)}."
            )
            raise ValueError(msg)
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive.")

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, RetrievalResult] = {}

    for list_idx, results in enumerate(result_lists):
        w = weights[list_idx] if weights is not None else 1.0
        for rank, result in enumerate(results):
            rrf_scores[result.chunk_id] = rrf_scores.get(
                result.chunk_id, 0.0,
            ) + (w / (k_rrf + rank + 1))

            if (
                result.chunk_id not in doc_map
                or result.score > doc_map[result.chunk_id].score
            ):
                doc_map[result.chunk_id] = result

    # Primary key: descending RRF score. Secondary key: ascending chunk_id.
    # The secondary key makes the output order deterministic across runs
    # when two documents tie on fused score (e.g. both reach rank #1 in
    # one list and are absent from the others).
    sorted_ids = sorted(
        rrf_scores,
        key=lambda cid: (-rrf_scores[cid], cid),
    )[:k]

    return [
        RetrievalResult(
            chunk_id=cid,
            text=doc_map[cid].text,
            score=rrf_scores[cid],
            metadata=doc_map[cid].metadata,
        )
        for cid in sorted_ids
    ]
