"""Data classes for the abstention stage of the MARley pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AbstentionResult:
    """Result of the abstention-aware pipeline.

    Captures whether the system abstained, at which level, and why.
    Level 1 abstention occurs when retrieval confidence is below the
    threshold. Level 2 abstention occurs when the LLM explicitly
    signals that the context is insufficient.
    """

    abstained: bool
    level: int | None       # 1 = retrieval confidence, 2 = LLM detection, None = answered
    reason: str             # Abstention reason (empty string if answered)
    answer: str             # Generated answer (empty string if abstained)
    confidence: float       # Top-1 normalized retrieval score
    retrieval_results: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
