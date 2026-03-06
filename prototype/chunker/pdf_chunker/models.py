from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkQualityFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    document_id: str
    source_file: str
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


class PDFChunk(BaseModel):
    chunk_id: str
    chunk_type: Literal["text", "table"]
    text: str
    token_count: int
    metadata: ChunkMetadata
    flags: list[ChunkQualityFlag] = Field(default_factory=list)


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
                total_chunks=0,
                text_chunks=text_chunks,
                table_chunks=table_chunks,
                sections_processed=sections_processed,
                sections_skipped=sections_skipped,
                tables_processed=tables_processed,
                tables_skipped=tables_skipped,
                min_tokens=0,
                median_tokens=0,
                max_tokens=0,
                total_tokens=0,
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


class ChunkingResult(BaseModel):
    document_id: str
    input_file: str
    source_file: str
    context: dict[str, Any]
    chunks: list[PDFChunk]
    stats: ChunkingStats
    quality_flags: list[ChunkQualityFlag] = Field(default_factory=list)
    created_at: datetime
    chunker_version: str = "0.1.0"


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
