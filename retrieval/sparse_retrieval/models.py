from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from retrieval.base import SearchHit as BaseSearchHit
from retrieval.base import SearchResponse as BaseSearchResponse

# Source types specific to our domain
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


class SearchHit(BaseSearchHit):
    rank: int
    # content is in BaseSearchHit (mapped from text)
    # metadata is in BaseSearchHit
    chunk_type: str
    input_file: str


class IndexRebuildRequest(BaseModel):
    document_id: str | None = None


class IndexRebuildResponse(BaseModel):
    document_id: str
    index_stats: IndexStats
    quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)


# We alias SearchRequest from base if we want strict compatibility, 
# or we define a compatible one. The sparse service expects specific fields.
# But for BaseRetriever compatibility, we should accept BaseSearchRequest structure.
# Let's keep a local SearchRequest that is compatible, or just use arguments in search() method.
# In Pydantic, inheritance handles field supersets.

class SearchRequest(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    source_types: list[SourceType] | None = None
    rebuild_if_stale: bool | None = None


class SearchResponse(BaseSearchResponse):
    document_id: str
    query: str
    top_k: int
    hits: list[SearchHit]
    index_stats: IndexStats
    quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)
