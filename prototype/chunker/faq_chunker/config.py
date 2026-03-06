from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FAQChunkerSettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    document_id: str = "msc-computer-science"
    data_root: Path = Path("data")
    input_dir: Path | None = None
    output_dir: Path | None = None

    input_glob: str = "*faq*.json"
    output_suffix: str = "faq-chunker"
    keep_previous_outputs: bool = True
    json_indent: int = Field(default=2, ge=0)

    tokenizer_encoding: str = "cl100k_base"
    faq_source: str = "so"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FAQ_CHUNKER_",
        extra="ignore",
    )

    @field_validator("faq_source")
    @classmethod
    def validate_faq_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("faq_source must not be empty")
        if re.fullmatch(r"[a-z0-9_]+", normalized) is None:
            raise ValueError("faq_source must match [a-z0-9_]+")
        return normalized

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
class FAQChunkerRuntimeConfig:
    input_glob: str
    output_suffix: str
    keep_previous_outputs: bool
    json_indent: int
    tokenizer_encoding: str
    faq_source: str
    input_dir: Path
    output_dir: Path


@lru_cache
def get_faq_chunker_settings() -> FAQChunkerSettings:
    return FAQChunkerSettings()


def get_faq_chunker_config(
    settings: FAQChunkerSettings | None = None,
) -> FAQChunkerRuntimeConfig:
    cfg = settings or get_faq_chunker_settings()
    return FAQChunkerRuntimeConfig(
        input_glob=cfg.input_glob,
        output_suffix=cfg.output_suffix,
        keep_previous_outputs=cfg.keep_previous_outputs,
        json_indent=cfg.json_indent,
        tokenizer_encoding=cfg.tokenizer_encoding,
        faq_source=cfg.faq_source,
        input_dir=cfg.input_dir_path,
        output_dir=cfg.output_dir_path,
    )
