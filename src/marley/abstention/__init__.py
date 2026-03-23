"""Abstention detection for the MARley system.

Provides LLM-level abstention detection via structured output analysis.
Score normalization and the pipeline orchestrator have moved to their
canonical locations in ``models.scoring`` and ``server.pipeline``.
"""

from src.marley.abstention.detection import (
    ABSTENTION_PREFIX,
    detect_abstention,
    extract_abstention_reason,
)

__all__ = [
    "ABSTENTION_PREFIX",
    "detect_abstention",
    "extract_abstention_reason",
]
