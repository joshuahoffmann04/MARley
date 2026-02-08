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


class FAQChunkMetadata(BaseModel):
    document_id: str
    source_file: str
    faq_source: str
    faq_id: str
    section: str | None = None
    paragraph_label: str | None = None
    page: int | None = None
    chunk_index: int = 0
    previous_question_id: str | None = None
    next_question_id: str | None = None
    previous_question_chunk_id: str | None = None
    next_question_chunk_id: str | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


class FAQChunk(BaseModel):
    chunk_id: str
    chunk_type: Literal["faq"] = "faq"
    text: str
    token_count: int
    metadata: FAQChunkMetadata
    flags: list[ChunkQualityFlag] = Field(default_factory=list)


class ChunkingStats(BaseModel):
    total_chunks: int = 0
    questions_total: int = 0
    questions_processed: int = 0
    questions_skipped: int = 0
    min_tokens: int = 0
    median_tokens: int = 0
    max_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_token_counts(
        cls,
        token_counts: list[int],
        *,
        questions_total: int,
        questions_processed: int,
        questions_skipped: int,
    ) -> "ChunkingStats":
        if not token_counts:
            return cls(
                total_chunks=0,
                questions_total=questions_total,
                questions_processed=questions_processed,
                questions_skipped=questions_skipped,
                min_tokens=0,
                median_tokens=0,
                max_tokens=0,
                total_tokens=0,
            )

        return cls(
            total_chunks=len(token_counts),
            questions_total=questions_total,
            questions_processed=questions_processed,
            questions_skipped=questions_skipped,
            min_tokens=min(token_counts),
            median_tokens=int(median(token_counts)),
            max_tokens=max(token_counts),
            total_tokens=sum(token_counts),
        )


class ChunkingResult(BaseModel):
    document_id: str
    faq_source: str
    input_file: str
    source_file: str
    context: dict[str, Any]
    chunks: list[FAQChunk]
    stats: ChunkingStats
    quality_flags: list[ChunkQualityFlag] = Field(default_factory=list)
    created_at: datetime
    chunker_version: str = "0.1.0"


class ChunkingRequest(BaseModel):
    input_file: str | None = None
    document_id: str | None = None
    output_dir: str | None = None
    faq_source: str | None = None
    persist_result: bool = True


class ChunkingResponse(BaseModel):
    document_id: str
    faq_source: str
    input_file: str
    output_file: str | None = None
    result: ChunkingResult
