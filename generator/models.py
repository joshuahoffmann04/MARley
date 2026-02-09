from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from retrieval.base import RetrievalQualityFlag, SourceType


class GenerationQualityFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    rank: int
    rrf_score: float
    source_type: SourceType
    chunk_id: str
    chunk_type: str
    text: str
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_file: str

    sparse_rank: int | None = None
    sparse_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    vector_distance: float | None = None


class RetrievalSearchResponse(BaseModel):
    document_id: str
    query: str
    top_k: int
    hits: list[RetrievalHit]
    quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)


class AbstentionOverrides(BaseModel):
    min_hits: int | None = Field(default=None, ge=1)
    min_best_rrf_score: float | None = Field(default=None, ge=0.0)
    min_dual_backend_hits: int | None = Field(default=None, ge=0)
    abstain_on_retrieval_errors: bool | None = None
    abstain_on_backend_degradation: bool | None = None


class GenerateRequest(BaseModel):
    query: str
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    source_types: list[SourceType] | None = None
    retrieval_rebuild_if_stale: bool | None = None

    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    total_budget_tokens: int | None = Field(default=None, ge=256, le=32768)
    max_answer_tokens: int | None = Field(default=None, ge=16, le=8192)

    include_used_chunks: bool = True
    abstention: AbstentionOverrides | None = None


class LLMStructuredOutput(BaseModel):
    answer: str
    should_abstain: bool
    confidence: Literal["low", "medium", "high"]
    reasoning: str


class UsedChunk(BaseModel):
    rank: int
    source_type: SourceType
    chunk_id: str
    chunk_type: str

    text: str
    token_count_original: int | None = None
    token_count_used: int
    truncated: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)
    input_file: str

    rrf_score: float
    sparse_rank: int | None = None
    sparse_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    vector_distance: float | None = None


class GenerateResponse(BaseModel):
    document_id: str
    query: str

    answer: str
    abstained: bool
    abstention_reason: str | None = None

    model: str
    temperature: float

    top_k: int
    retrieval_hit_count: int

    total_budget_tokens: int
    max_answer_tokens: int
    prompt_overhead_tokens: int
    context_budget_tokens: int
    used_context_tokens: int

    used_chunks: list[UsedChunk] = Field(default_factory=list)

    retrieval_quality_flags: list[RetrievalQualityFlag] = Field(default_factory=list)
    generator_quality_flags: list[GenerationQualityFlag] = Field(default_factory=list)

    generated_at: datetime
