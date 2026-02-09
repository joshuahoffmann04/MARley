from __future__ import annotations

from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, Field

from chunker.base import BaseChunk, BaseChunkMetadata, ChunkingResult as BaseChunkingResult
from chunker.base import ChunkQualityFlag

# Re-export ChunkQualityFlag for backward compatibility
__all__ = [
    "ChunkQualityFlag",
    "ChunkMetadata",
    "PDFChunk",
    "ChunkingStats",
    "ChunkingResult",
    "ChunkingRequest",
    "ChunkingResponse",
]


class ChunkMetadata(BaseChunkMetadata):
    section_id: str | None = None
    section_kind: Literal["part", "paragraph", "annex"] | None = None
    section_label: str | None = None
    section_title: str | None = None
    parent_section_id: str | None = None
    path_labels: list[str] = Field(default_factory=list)
    start_page: int | None = None
    end_page: int | None = None
    table_ids: list[str] = Field(default_factory=list)
    table_id: str | None = None
    chunk_index_in_section: int = 0
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


class PDFChunk(BaseChunk):
    chunk_type: Literal["text", "table"]
    metadata: ChunkMetadata


class ChunkingStats(BaseModel):
    total_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    sections_processed: int = 0
    sections_skipped: int = 0
    tables_processed: int = 0
    tables_skipped: int = 0
    min_tokens: int = 0
    median_tokens: int = 0
    max_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_token_counts(
        cls,
        token_counts: list[int],
        *,
        text_chunks: int,
        table_chunks: int,
        sections_processed: int,
        sections_skipped: int,
        tables_processed: int,
        tables_skipped: int,
    ) -> "ChunkingStats":
        if not token_counts:
            return cls(
                text_chunks=text_chunks,
                table_chunks=table_chunks,
                sections_processed=sections_processed,
                sections_skipped=sections_skipped,
                tables_processed=tables_processed,
                tables_skipped=tables_skipped,
            )

        return cls(
            total_chunks=len(token_counts),
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            sections_processed=sections_processed,
            sections_skipped=sections_skipped,
            tables_processed=tables_processed,
            tables_skipped=tables_skipped,
            min_tokens=min(token_counts),
            median_tokens=int(median(token_counts)),
            max_tokens=max(token_counts),
            total_tokens=sum(token_counts),
        )


class ChunkingResult(BaseChunkingResult):
    context: dict[str, Any]
    chunks: list[PDFChunk]
    stats: ChunkingStats
    chunker_version: str = "0.2.0"


class ChunkingRequest(BaseModel):
    input_file: str | None = None
    document_id: str | None = None
    output_dir: str | None = None
    persist_result: bool = True


class ChunkingResponse(BaseModel):
    document_id: str
    input_file: str
    output_file: str | None = None
    result: ChunkingResult
