"""Re-export pipeline orchestrator from its canonical location.

The authoritative definition lives in ``src.marley.server.pipeline``.
This module re-exports it for backward compatibility.
"""

from src.marley.server.pipeline import run_with_abstention

__all__ = ["run_with_abstention"]
