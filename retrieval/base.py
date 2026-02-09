from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel, Field

SourceType = Literal["pdf", "faq_so", "faq_sb"]


class RetrievalQualityFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    document_id: str
    top_k: int = 5
    source_types: list[str] | None = None
    rebuild_if_stale: bool | None = None


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    content: str
    metadata: dict
    source_type: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    total_found: int = 0
    processing_time_ms: float = 0.0


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search."""
        pass

    @abstractmethod
    def index(self, document_id: str, rebuild: bool = False) -> None:
        """Build or update index."""
        pass
