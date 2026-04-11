"""Server configuration for the MARley pipeline.

Defines paths, defaults, and the Ollama connectivity check.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.marley.models.constants import (
    DEFAULT_K,
    DEFAULT_THRESHOLDS,
    NORMALIZATION_MAP,
    RETRIEVER_TYPES,
    STRATEGIES,
)

# Default chunk file paths (relative to project root)
CHUNK_PATHS: dict[str, str] = {
    "stpo": "data/chunks/stpo-chunks.json",
    "faq-stpo": "data/chunks/faq-stpo-chunks.json",
    "faq-ao": "data/chunks/faq-ao-chunks.json",
}


@dataclass
class ServerConfig:
    """Runtime configuration for the MARley server."""

    host: str = "127.0.0.1"
    port: int = 8000
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    chunk_dir: Path = field(default_factory=lambda: Path("data/chunks"))
    pdf_path: Path | None = field(default_factory=lambda: Path("data/raw/msc-computer-science.pdf"))
    k: int = DEFAULT_K
    default_retriever_type: str = "hybrid"
    default_strategy: str = "merged_pool"
    default_knowledge_bases: list[str] = field(
        default_factory=lambda: ["stpo", "faq-stpo", "faq-ao"],
    )


def check_ollama(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Check Ollama server connectivity.

    Returns:
        Dict with 'available' (bool) and optional 'error' (str).
    """
    request = urllib.request.Request(
        url=f"{base_url}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "available": 200 <= response.status < 300,
                "status_code": response.status,
            }
    except urllib.error.URLError as exc:
        return {"available": False, "error": str(exc.reason)}
    except socket.timeout:
        return {"available": False, "error": "request timed out"}
