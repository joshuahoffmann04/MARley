"""Shared quality flag for pipeline diagnostics.

Raised by extractors, chunkers, and other pipeline stages to record
non-fatal issues encountered during processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityFlag:
    """A diagnostic flag raised during pipeline processing.

    Attributes:
        code: Machine-readable identifier (e.g. ``"EMPTY_SECTION"``).
        message: Human-readable description of the issue.
        severity: One of ``"info"``, ``"warning"``, ``"error"``.
        context: Arbitrary key-value pairs providing additional detail.
    """

    code: str
    message: str
    severity: str
    context: dict[str, Any] = field(default_factory=dict)
