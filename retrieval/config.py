from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from MARley.config import Settings, get_settings


@dataclass(frozen=True)
class PDFExtractorRuntimeConfig:
    output_suffix: str
    keep_previous_outputs: bool
    knowledgebases_dir: Path
    raw_dir: Path


def get_pdf_extractor_config(settings: Settings | None = None) -> PDFExtractorRuntimeConfig:
    cfg = settings or get_settings()
    return PDFExtractorRuntimeConfig(
        output_suffix=cfg.pdf_extractor_output_suffix,
        keep_previous_outputs=cfg.pdf_extractor_keep_previous_outputs,
        knowledgebases_dir=cfg.knowledgebases_dir,
        raw_dir=cfg.raw_dir,
    )
