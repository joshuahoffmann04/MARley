from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SparseRetrievalSettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    document_id: str = "msc-computer-science"
    data_root: Path = Path("data")
    input_dir: Path | None = None

    pdf_chunks_glob: str = "*-pdf-chunker-*.json"
    faq_so_chunks_glob: str = "*-faq-chunker-so-*.json"
    faq_sb_chunks_glob: str = "*-faq-chunker-sb-*.json"

    top_k_default: int = Field(default=10, ge=1)
    top_k_max: int = Field(default=100, ge=1)

    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    bm25_epsilon: float = Field(default=0.25, ge=0.0)

    lowercase: bool = True
    remove_stopwords: bool = True
    min_token_length: int = Field(default=2, ge=1)
    include_numeric_tokens: bool = True

    auto_rebuild_on_search: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SPARSE_RETRIEVAL_",
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
        return (self.document_root / "chunks").resolve()


@dataclass(frozen=True)
class SparseRetrievalRuntimeConfig:
    input_dir: Path
    pdf_chunks_glob: str
    faq_so_chunks_glob: str
    faq_sb_chunks_glob: str
    top_k_default: int
    top_k_max: int
    bm25_k1: float
    bm25_b: float
    bm25_epsilon: float
    lowercase: bool
    remove_stopwords: bool
    min_token_length: int
    include_numeric_tokens: bool
    auto_rebuild_on_search: bool


@lru_cache
def get_sparse_retrieval_settings() -> SparseRetrievalSettings:
    return SparseRetrievalSettings()


def get_sparse_retrieval_config(
    settings: SparseRetrievalSettings | None = None,
) -> SparseRetrievalRuntimeConfig:
    cfg = settings or get_sparse_retrieval_settings()
    return SparseRetrievalRuntimeConfig(
        input_dir=cfg.input_dir_path,
        pdf_chunks_glob=cfg.pdf_chunks_glob,
        faq_so_chunks_glob=cfg.faq_so_chunks_glob,
        faq_sb_chunks_glob=cfg.faq_sb_chunks_glob,
        top_k_default=cfg.top_k_default,
        top_k_max=cfg.top_k_max,
        bm25_k1=cfg.bm25_k1,
        bm25_b=cfg.bm25_b,
        bm25_epsilon=cfg.bm25_epsilon,
        lowercase=cfg.lowercase,
        remove_stopwords=cfg.remove_stopwords,
        min_token_length=cfg.min_token_length,
        include_numeric_tokens=cfg.include_numeric_tokens,
        auto_rebuild_on_search=cfg.auto_rebuild_on_search,
    )
