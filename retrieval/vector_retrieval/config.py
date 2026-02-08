from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorRetrievalSettings(BaseSettings):
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

    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"
    chroma_collection_prefix: str = "marley-vector"
    chroma_distance_metric: str = "cosine"

    ollama_embedding_url: str = "http://localhost:11434/api/embeddings"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    embedding_batch_size: int = Field(default=64, ge=1)

    auto_rebuild_on_search: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VECTOR_RETRIEVAL_",
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
class VectorRetrievalRuntimeConfig:
    input_dir: Path
    pdf_chunks_glob: str
    faq_so_chunks_glob: str
    faq_sb_chunks_glob: str
    top_k_default: int
    top_k_max: int
    chroma_host: str
    chroma_port: int
    chroma_ssl: bool
    chroma_tenant: str
    chroma_database: str
    chroma_collection_prefix: str
    chroma_distance_metric: str
    ollama_embedding_url: str
    ollama_embedding_model: str
    embedding_batch_size: int
    auto_rebuild_on_search: bool


@lru_cache
def get_vector_retrieval_settings() -> VectorRetrievalSettings:
    return VectorRetrievalSettings()


def get_vector_retrieval_config(
    settings: VectorRetrievalSettings | None = None,
) -> VectorRetrievalRuntimeConfig:
    cfg = settings or get_vector_retrieval_settings()
    return VectorRetrievalRuntimeConfig(
        input_dir=cfg.input_dir_path,
        pdf_chunks_glob=cfg.pdf_chunks_glob,
        faq_so_chunks_glob=cfg.faq_so_chunks_glob,
        faq_sb_chunks_glob=cfg.faq_sb_chunks_glob,
        top_k_default=cfg.top_k_default,
        top_k_max=cfg.top_k_max,
        chroma_host=cfg.chroma_host,
        chroma_port=cfg.chroma_port,
        chroma_ssl=cfg.chroma_ssl,
        chroma_tenant=cfg.chroma_tenant,
        chroma_database=cfg.chroma_database,
        chroma_collection_prefix=cfg.chroma_collection_prefix,
        chroma_distance_metric=cfg.chroma_distance_metric,
        ollama_embedding_url=cfg.ollama_embedding_url,
        ollama_embedding_model=cfg.ollama_embedding_model,
        embedding_batch_size=cfg.embedding_batch_size,
        auto_rebuild_on_search=cfg.auto_rebuild_on_search,
    )
