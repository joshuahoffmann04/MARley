"""Re-export abstention metrics from their canonical location.

The authoritative definitions live in ``evaluation.utils``.
This module re-exports them for backward compatibility.
"""

from evaluation.utils import AbstentionMetrics, compute_abstention_metrics

__all__ = ["AbstentionMetrics", "compute_abstention_metrics"]
