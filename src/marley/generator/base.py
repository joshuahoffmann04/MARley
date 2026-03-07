"""Abstract base class for generation strategies.

All generators in the MARley pipeline implement the Generator interface,
ensuring consistent behavior across different LLM backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.marley.models.generation import GenerationResult


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
