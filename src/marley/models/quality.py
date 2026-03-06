"""Shared quality flag for pipeline diagnostics.

Raised by extractors, chunkers, and other pipeline stages to record
non-fatal issues encountered during processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityFlag:
    """A diagnostic flag raised during pipeline processing."""
    code: str
    message: str
    severity: str  # "info", "warning", "error"
    context: dict = field(default_factory=dict)
