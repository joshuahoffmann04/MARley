from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "faq_so", "faq_sb"]


class RetrievalQualityFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class SourceSnapshot(BaseModel):
    source_type: SourceType
    file_path: str | None = None
    chunk_count: int = 0
    missing: bool = False
    last_modified: datetime | None = None


class IndexStats(BaseModel):
    document_id: str
    total_chunks: int
    chunks_by_source: dict[str, int]
    built_at: datetime
    index_signature: str
    source_snapshots: list[SourceSnapshot]


class SearchHit(BaseModel):
    rank: int
    score: float
    source_type: SourceType
    chunk_id: str
    chunk_type: str
    text: str
    token_count: int | None = None
    metadata: dict[str, Any]
    input_file: str


class IndexRebuildRequest(BaseModel):
    document_id: str | None = None


class IndexRebuildResponse(BaseModel):
    document_id: str
    index_stats: IndexStats
    quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    source_types: list[SourceType] | None = None
    rebuild_if_stale: bool | None = None


class SearchResponse(BaseModel):
    document_id: str
    query: str
    top_k: int
    hits: list[SearchHit]
    index_stats: IndexStats
    quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)
