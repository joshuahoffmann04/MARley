"""Generation data classes and abstract base for the MARley pipeline.

Defines the GenerationResult dataclass and the Generator abstract base
class.  All generator implementations import from this module to ensure
a single source of truth for the generation interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    """Result of a single generation call."""

    answer: str
    model: str
    context_chunk_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Generator(ABC):
    """Abstract base for all generation strategies."""

    @abstractmethod
    def generate(self, query: str, context: list[dict]) -> GenerationResult:
        """Generate an answer given a query and context chunks.

        Args:
            query: The user question.
            context: List of chunk dicts, each with 'chunk_id', 'text',
                and optionally 'metadata'.

        Returns:
            A GenerationResult with the generated answer.
        """
