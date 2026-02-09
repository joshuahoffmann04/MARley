from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkQualityFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class BaseChunkMetadata(BaseModel):
    document_id: str
    source_file: str
    chunk_index: int
    page: int | None = None
    section_id: str | None = None
    section_title: str | None = None
    # Add other common fields as needed


class BaseChunk(BaseModel):
    chunk_id: str
    chunk_type: str  # "text", "table", "faq"
    text: str
    token_count: int
    metadata: dict[str, Any] # Flexible dict for specific metadata
    flags: list[ChunkQualityFlag] = Field(default_factory=list)


class ChunkingResult(BaseModel):
    document_id: str
    input_file: str
    chunks: list[BaseChunk]
    quality_flags: list[ChunkQualityFlag] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self) -> ChunkingResult:
        """Execute chunking and return result."""
        pass
