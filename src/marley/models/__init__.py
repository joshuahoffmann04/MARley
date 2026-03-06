"""Shared data classes for the MARley pipeline."""

from src.marley.models.extraction import ExtractionResult, Section, Table
from src.marley.models.quality import QualityFlag

__all__ = [
    "ExtractionResult",
    "Section",
    "Table",
    "QualityFlag",
]
