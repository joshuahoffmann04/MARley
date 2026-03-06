from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["sparse", "vector", "hybrid"]


class MarleySettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    data_root: Path = Path("data")
    default_document_id: str = "msc-computer-science"
    default_retrieval_mode: RetrievalMode = "hybrid"

    top_k_default: int = Field(default=10, ge=1)
    top_k_max: int = Field(default=100, ge=1)

    ollama_base_url: str = "http://127.0.0.1:11434"
    generator_model: str = "llama3.1:latest"
    generator_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generator_total_budget_tokens: int = Field(default=2048, ge=256)
    generator_max_answer_tokens: int = Field(default=384, ge=16)

    embedding_model: str = "nomic-embed-text:latest"
    vector_persist_subdir: str = "databases"

    rrf_rank_constant: int = Field(default=60, ge=1)
    rank_window_size: int = Field(default=50, ge=1)
    sparse_weight: float = Field(default=1.0, gt=0.0)
    vector_weight: float = Field(default=1.0, gt=0.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MARLEY_",
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


@dataclass(frozen=True)
class MarleyRuntimeConfig:
    data_root: Path
    default_document_id: str
    default_retrieval_mode: RetrievalMode
    top_k_default: int
    top_k_max: int
    ollama_base_url: str
    generator_model: str
    generator_temperature: float
    generator_total_budget_tokens: int
    generator_max_answer_tokens: int
    embedding_model: str
    vector_persist_subdir: str
    rrf_rank_constant: int
    rank_window_size: int
    sparse_weight: float
    vector_weight: float


@lru_cache
def get_marley_settings() -> MarleySettings:
    return MarleySettings()


def get_marley_config(settings: MarleySettings | None = None) -> MarleyRuntimeConfig:
    cfg = settings or get_marley_settings()
    return MarleyRuntimeConfig(
        data_root=cfg.data_root_path,
        default_document_id=cfg.default_document_id,
        default_retrieval_mode=cfg.default_retrieval_mode,
        top_k_default=cfg.top_k_default,
        top_k_max=cfg.top_k_max,
        ollama_base_url=cfg.ollama_base_url.rstrip("/"),
        generator_model=cfg.generator_model,
        generator_temperature=cfg.generator_temperature,
        generator_total_budget_tokens=cfg.generator_total_budget_tokens,
        generator_max_answer_tokens=cfg.generator_max_answer_tokens,
        embedding_model=cfg.embedding_model,
        vector_persist_subdir=cfg.vector_persist_subdir,
        rrf_rank_constant=cfg.rrf_rank_constant,
        rank_window_size=cfg.rank_window_size,
        sparse_weight=cfg.sparse_weight,
        vector_weight=cfg.vector_weight,
    )
