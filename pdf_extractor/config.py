from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PDFExtractorSettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    document_id: str = "2-aend-19-02-25_msc-computer-science_lese"
    data_root: Path = Path("data")
    raw_dir: Path | None = None
    knowledgebases_dir: Path | None = None

    output_suffix: str = "pdf-extractor"
    keep_previous_outputs: bool = True
    json_indent: int = Field(default=2, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PDF_EXTRACTOR_",
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.project_root / value).resolve()

    @property
    def data_root_path(self) -> Path:
        return self.resolve_path(self.data_root)

    @property
    def document_root(self) -> Path:
        return (self.data_root_path / self.document_id).resolve()

    @property
    def raw_dir_path(self) -> Path:
        if self.raw_dir is not None:
            return self.resolve_path(self.raw_dir)
        return (self.document_root / "raw").resolve()

    @property
    def knowledgebases_dir_path(self) -> Path:
        if self.knowledgebases_dir is not None:
            return self.resolve_path(self.knowledgebases_dir)
        return (self.document_root / "knowledgebases").resolve()


@dataclass(frozen=True)
class PDFExtractorRuntimeConfig:
    output_suffix: str
    keep_previous_outputs: bool
    json_indent: int
    knowledgebases_dir: Path
    raw_dir: Path


@lru_cache
def get_pdf_extractor_settings() -> PDFExtractorSettings:
    return PDFExtractorSettings()


def get_pdf_extractor_config(
    settings: PDFExtractorSettings | None = None,
) -> PDFExtractorRuntimeConfig:
    cfg = settings or get_pdf_extractor_settings()
    return PDFExtractorRuntimeConfig(
        output_suffix=cfg.output_suffix,
        keep_previous_outputs=cfg.keep_previous_outputs,
        json_indent=cfg.json_indent,
        knowledgebases_dir=cfg.knowledgebases_dir_path,
        raw_dir=cfg.raw_dir_path,
    )
