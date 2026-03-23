"""Pydantic request and response models for the MARley server API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    query: str = Field(..., min_length=1, description="User question")
    retriever_type: str = Field(
        default="hybrid",
        description="Retriever type: bm25, vector, or hybrid",
    )
    knowledge_bases: list[str] = Field(
        default_factory=lambda: ["stpo", "faq-stpo", "faq-ao"],
        description="Knowledge bases to search",
    )
    strategy: str = Field(
        default="merged_pool",
        description="Combination strategy: single, merged_pool, or fusion",
    )
    k: int = Field(default=5, ge=1, le=50, description="Top-k retrieval count")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Abstention threshold override (None = auto)",
    )


class SourceReference(BaseModel):
    """A single source chunk referenced in the answer."""

    chunk_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class ChatConfigInfo(BaseModel):
    """Configuration metadata included in the response."""

    retriever_type: str
    knowledge_bases: list[str]
    strategy: str
    k: int
    threshold: float
    normalization_strategy: str
    model: str


class ChatResponse(BaseModel):
    """Response body for POST /api/chat."""

    answer: str
    abstained: bool
    abstention_level: int | None = None
    abstention_reason: str = ""
    confidence: float
    sources: list[SourceReference]
    config: ChatConfigInfo


class OptionsResponse(BaseModel):
    """Response for GET /api/options."""

    retriever_types: list[str]
    knowledge_bases: list[str]
    strategies: list[str]
    defaults: dict
    ollama_model: str
    ollama_status: str


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: str
    ollama: str
    model: str
    cached_retrievers: int
    knowledge_bases: list[str]
