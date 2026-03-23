"""Re-export scoring functions from their canonical location.

The authoritative definitions live in ``src.marley.models.scoring``.
This module re-exports them for backward compatibility.
"""

from src.marley.models.scoring import (
    compute_confidence,
    filter_by_threshold,
    normalize_scores,
)

__all__ = ["compute_confidence", "filter_by_threshold", "normalize_scores"]
