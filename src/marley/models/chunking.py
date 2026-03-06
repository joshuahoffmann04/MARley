"""Shared chunking utilities for the MARley pipeline.

Provides common helper functions used by both the PDF chunker and
the FAQ chunker to avoid code duplication.
"""

from __future__ import annotations

from statistics import median


def compute_token_stats(token_counts: list[int]) -> dict[str, int]:
    """Compute min, median, max, and total token statistics.

    Args:
        token_counts: List of token counts, one per chunk.

    Returns:
        Dictionary with keys min_tokens, median_tokens, max_tokens,
        total_tokens.  All values are 0 if the list is empty.
    """
    if not token_counts:
        return {
            "min_tokens": 0,
            "median_tokens": 0,
            "max_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "min_tokens": min(token_counts),
        "median_tokens": int(median(token_counts)),
        "max_tokens": max(token_counts),
        "total_tokens": sum(token_counts),
    }
