from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class QualityFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class DocumentContext(BaseModel):
    document_type: str | None = None
    title: str | None = None
    institution: str | None = None
    version_date: date | None = None
    faculty: str | None = None
    total_pages: int


class SectionRecord(BaseModel):
    section_id: str
    parent_section_id: str | None = None
    kind: Literal["section", "paragraph", "annex"]
    label: str
    title: str | None = None
    start_page: int
    end_page: int
    text: str
    table_ids: list[str] = Field(default_factory=list)
    flags: list[QualityFlag] = Field(default_factory=list)


class TableRecord(BaseModel):
    table_id: str
    page: int
    section_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    rows: list[list[str]]
    flags: list[QualityFlag] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    document_id: str
    source_file: str
    context: DocumentContext
    sections: list[SectionRecord]
    tables: list[TableRecord]
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    created_at: datetime


class ExtractionRequest(BaseModel):
    source_file: str | None = None
    document_id: str | None = None
    output_dir: str | None = None


class ExtractionJob(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    document_id: str
    source_file: str
    output_file: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    section_count: int = 0
    table_count: int = 0
    error: str | None = None


class ExtractionResponse(BaseModel):
    job: ExtractionJob

