"""Re-export generation base classes from their canonical location.

The authoritative definitions live in ``src.marley.models.generation``.
This module re-exports them so that generator implementations can use
short import paths.
"""

from src.marley.models.generation import Generator, GenerationResult

__all__ = ["Generator", "GenerationResult"]
