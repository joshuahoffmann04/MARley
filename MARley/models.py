from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from generator.models import SourceType

RetrievalMode = Literal["sparse", "vector", "hybrid"]


class ChatRequest(BaseModel):
    query: str
    document_id: str
    retrieval_mode: RetrievalMode
    source_types: list[SourceType] | None = None


class DocumentSourceAvailability(BaseModel):
    source_type: SourceType
    available: bool
    latest_file: str | None = None
    last_modified: datetime | None = None


class DocumentOption(BaseModel):
    document_id: str
    sources: list[DocumentSourceAvailability]


class OptionsResponse(BaseModel):
    documents: list[DocumentOption]
    default_document_id: str | None = None
    default_retrieval_mode: RetrievalMode = "hybrid"


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    document_id: str
    retrieval_mode: RetrievalMode
    source_types: list[SourceType]
    generated_at: datetime

    generator_response: dict[str, Any] = Field(default_factory=dict)
    retrieval_response: dict[str, Any] = Field(default_factory=dict)
