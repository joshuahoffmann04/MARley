"""Shared I/O utilities for the MARley pipeline.

Provides a generic JSON serialization function used by all pipeline
stages (extractor, chunkers) to persist dataclass results to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def save_json(result: Any, output_path: str | Path) -> Path:
    """Serialize a dataclass instance to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        result: A dataclass instance to serialize via ``dataclasses.asdict``.
        output_path: Destination file path.

    Returns:
        The resolved absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path.resolve()
