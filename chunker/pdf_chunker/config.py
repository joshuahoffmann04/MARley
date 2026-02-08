from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PDFChunkerSettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    document_id: str = "msc-computer-science"
    data_root: Path = Path("data")
    input_dir: Path | None = None
    output_dir: Path | None = None

    input_glob: str = "*-pdf-extractor-*.json"
    output_suffix: str = "pdf-chunker"
    keep_previous_outputs: bool = True
    json_indent: int = Field(default=2, ge=0)

    tokenizer_encoding: str = "cl100k_base"
    min_chunk_tokens: int = Field(default=256, ge=1)
    max_chunk_tokens: int = Field(default=512, ge=16)
    overlap_tokens: int = Field(default=64, ge=0)

    include_table_chunks: bool = True
    prepend_heading_path: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PDF_CHUNKER_",
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.project_root / value).resolve()

    @property
    def data_root_path(self) -> Path:
        return self.resolve_path(self.data_root)

    @property
    def document_root(self) -> Path:
        return (self.data_root_path / self.document_id).resolve()

    @property
    def input_dir_path(self) -> Path:
        if self.input_dir is not None:
            return self.resolve_path(self.input_dir)
        return (self.document_root / "knowledgebases").resolve()

    @property
    def output_dir_path(self) -> Path:
        if self.output_dir is not None:
            return self.resolve_path(self.output_dir)
        return (self.document_root / "chunks").resolve()


@dataclass(frozen=True)
class PDFChunkerRuntimeConfig:
    input_glob: str
    output_suffix: str
    keep_previous_outputs: bool
    json_indent: int
    tokenizer_encoding: str
    min_chunk_tokens: int
    max_chunk_tokens: int
    overlap_tokens: int
    include_table_chunks: bool
    prepend_heading_path: bool
    input_dir: Path
    output_dir: Path


@lru_cache
def get_pdf_chunker_settings() -> PDFChunkerSettings:
    settings = PDFChunkerSettings()
    if settings.min_chunk_tokens > settings.max_chunk_tokens:
        raise ValueError("min_chunk_tokens must be <= max_chunk_tokens")
    if settings.overlap_tokens >= settings.max_chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")
    return settings


def get_pdf_chunker_config(
    settings: PDFChunkerSettings | None = None,
) -> PDFChunkerRuntimeConfig:
    cfg = settings or get_pdf_chunker_settings()
    return PDFChunkerRuntimeConfig(
        input_glob=cfg.input_glob,
        output_suffix=cfg.output_suffix,
        keep_previous_outputs=cfg.keep_previous_outputs,
        json_indent=cfg.json_indent,
        tokenizer_encoding=cfg.tokenizer_encoding,
        min_chunk_tokens=cfg.min_chunk_tokens,
        max_chunk_tokens=cfg.max_chunk_tokens,
        overlap_tokens=cfg.overlap_tokens,
        include_table_chunks=cfg.include_table_chunks,
        prepend_heading_path=cfg.prepend_heading_path,
        input_dir=cfg.input_dir_path,
        output_dir=cfg.output_dir_path,
    )
