"""Shared data classes and utilities for the MARley pipeline."""

from src.marley.models.abstention import AbstentionResult
from src.marley.models.chunking import compute_token_stats
from src.marley.models.extraction import ExtractionResult, Section, Table
from src.marley.models.generation import GenerationResult, Generator
from src.marley.models.io import save_json
from src.marley.models.quality import QualityFlag
from src.marley.models.retrieval import (
    RetrievalResult,
    Retriever,
    load_chunks,
    rrf_fuse,
    validate_corpus,
)
from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)

__all__ = [
    "AbstentionResult",
    "ExtractionResult",
    "GenerationResult",
    "Generator",
    "QualityFlag",
    "RetrievalResult",
    "Retriever",
    "Section",
    "Table",
    "compute_confidence",
    "compute_token_stats",
    "filter_by_threshold",
    "load_chunks",
    "normalize_scores",
    "rrf_fuse",
    "save_json",
    "validate_corpus",
]
