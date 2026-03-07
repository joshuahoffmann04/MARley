"""Data classes for the generation stage of the MARley pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    """Result of a single generation call."""

    answer: str
    model: str
    context_chunk_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
