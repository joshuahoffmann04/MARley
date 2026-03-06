from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "faq_so", "faq_sb"]
BackendName = Literal["sparse", "vector"]


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


class BackendIndexStats(BaseModel):
    backend: BackendName
    available: bool = False
    status_code: int | None = None
    error: str | None = None
    total_chunks: int | None = None
    chunks_by_source: dict[str, int] = Field(default_factory=dict)
    built_at: datetime | None = None
    index_signature: str | None = None
    collection_name: str | None = None
    source_snapshots: list[SourceSnapshot] = Field(default_factory=list)


class HybridIndexStats(BaseModel):
    document_id: str
    built_at: datetime
    index_signature: str
    fused_from_backends: list[BackendName] = Field(default_factory=list)
    sparse: BackendIndexStats
    vector: BackendIndexStats


class HybridSearchHit(BaseModel):
    rank: int
    rrf_score: float
    source_type: SourceType
    chunk_id: str
    chunk_type: str
    text: str
    token_count: int | None = None
    metadata: dict[str, Any]
    input_file: str

    sparse_rank: int | None = None
    sparse_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    vector_distance: float | None = None


class IndexRebuildRequest(BaseModel):
    document_id: str | None = None


class IndexRebuildResponse(BaseModel):
    document_id: str
    index_stats: HybridIndexStats
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
    hits: list[HybridSearchHit]
    index_stats: HybridIndexStats
    quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)
