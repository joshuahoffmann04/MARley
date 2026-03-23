"""MARley evaluation harness.

Unified evaluation framework covering retrieval, generation,
abstention, and end-to-end pipeline evaluation.

CLI entry point::

    python -m evaluation --help
"""

from evaluation.validate import validate_data_requirements

__all__ = [
    "validate_data_requirements",
]
