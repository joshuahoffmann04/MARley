"""Shared constants for the MARley pipeline.

Centralizes configuration defaults and enumerations used across
multiple modules to avoid duplication and ensure consistency.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_K: int = 5
"""Default number of chunks to retrieve."""

DEFAULT_K_RRF: int = 1
"""General default RRF smoothing constant.

Set to 1 based on the empirical RRF sweep: on all three knowledge bases
(stpo, faq-stpo, faq-ao) and for both Hybrid (BM25+Vector) and Fusion
(cross-KB), k_rrf = 1 maximises F1@5. Lower k_rrf sharpens RRF towards
a max-join, which rewards strong top-1 agreement more than the classic
IR-literature default of 60.
"""

DEFAULT_K_RRF_HYBRID: int = 1
"""Default RRF smoothing constant for HybridRetriever (BM25 + Vector).
See :data:`DEFAULT_K_RRF` for the empirical justification."""

DEFAULT_K_RRF_FUSION: int = 1
"""Default RRF smoothing constant for FusionRetriever (cross-KB).
See :data:`DEFAULT_K_RRF` for the empirical justification."""

RETRIEVER_TYPES: list[str] = ["bm25", "vector", "hybrid"]
"""Supported retriever type identifiers."""

STRATEGIES: list[str] = ["single", "merged_pool", "fusion"]
"""Supported knowledge-base combination strategies."""

# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

NORMALIZATION_STRATEGIES: set[str] = {"bm25", "vector", "rrf"}
"""Recognized normalization strategy names."""

NORMALIZATION_MAP: dict[str, str] = {
    "bm25": "bm25",
    "vector": "vector",
    "hybrid": "rrf",
}
"""Mapping from retriever type to its normalization strategy."""

DEFAULT_THRESHOLD: float = 0.3
"""Default abstention confidence threshold."""

DEFAULT_THRESHOLDS: dict[str, float] = {
    "bm25": DEFAULT_THRESHOLD,
    "vector": DEFAULT_THRESHOLD,
    "rrf": DEFAULT_THRESHOLD,
}
"""Per-strategy default thresholds (currently identical)."""

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

DEFAULT_MAX_CHUNK_TOKENS: int = 512
"""Maximum token count per chunk."""

DEFAULT_MIN_CHUNK_TOKENS: int = 64
"""Minimum token count per chunk (smaller chunks are merged)."""

DEFAULT_OVERLAP_TOKENS: int = 50
"""Token overlap between consecutive sliding-window chunks."""

DEFAULT_TOKENIZER: str = "cl100k_base"
"""Default tiktoken encoding for token counting."""

# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

CHROMADB_BATCH_SIZE: int = 5000
"""Maximum batch size for ChromaDB add operations."""

# ---------------------------------------------------------------------------
# Server / display
# ---------------------------------------------------------------------------

SOURCE_TEXT_TRUNCATION: int = 500
"""Maximum character length for source text snippets in API responses."""
