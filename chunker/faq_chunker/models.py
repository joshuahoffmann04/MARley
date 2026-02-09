from __future__ import annotations

from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, Field

from chunker.base import BaseChunk, BaseChunkMetadata, ChunkingResult as BaseChunkingResult
from chunker.base import ChunkQualityFlag

__all__ = [
    "ChunkQualityFlag",
    "FAQChunkMetadata",
    "FAQChunk",
    "ChunkingStats",
    "ChunkingResult",
    "ChunkingRequest",
    "ChunkingResponse",
]


class FAQChunkMetadata(BaseChunkMetadata):
    faq_source: str
    faq_id: str
    section: str | None = None
    paragraph_label: str | None = None
    page: int | None = None
    previous_question_id: str | None = None
    next_question_id: str | None = None
    previous_question_chunk_id: str | None = None
    next_question_chunk_id: str | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


class FAQChunk(BaseChunk):
    chunk_type: Literal["faq"] = "faq"
    metadata: FAQChunkMetadata


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
                questions_total=questions_total,
                questions_processed=questions_processed,
                questions_skipped=questions_skipped,
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


class ChunkingResult(BaseChunkingResult):
    faq_source: str
    context: dict[str, Any]
    chunks: list[FAQChunk]
    stats: ChunkingStats
    chunker_version: str = "0.2.0"


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
