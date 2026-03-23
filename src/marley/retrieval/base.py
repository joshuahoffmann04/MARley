"""Re-export retrieval base classes from their canonical location.

The authoritative definitions live in ``src.marley.models.retrieval``.
This module re-exports them so that internal retriever implementations
can use short import paths.
"""

from src.marley.models.retrieval import RetrievalResult, Retriever

__all__ = ["RetrievalResult", "Retriever"]
