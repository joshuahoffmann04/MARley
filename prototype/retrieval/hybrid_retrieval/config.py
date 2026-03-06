from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HybridRetrievalSettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    document_id: str = "msc-computer-science"

    top_k_default: int = Field(default=10, ge=1)
    top_k_max: int = Field(default=100, ge=1)

    rrf_rank_constant: int = Field(default=60, ge=1)
    rank_window_size: int = Field(default=50, ge=1)
    sparse_weight: float = Field(default=1.0, gt=0.0)
    vector_weight: float = Field(default=1.0, gt=0.0)

    sparse_base_url: str = "http://127.0.0.1:8004"
    vector_base_url: str = "http://127.0.0.1:8005"
    http_timeout_seconds: float = Field(default=30.0, gt=0.0)

    auto_rebuild_on_search: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="HYBRID_RETRIEVAL_",
        extra="ignore",
    )


@dataclass(frozen=True)
class HybridRetrievalRuntimeConfig:
    top_k_default: int
    top_k_max: int
    rrf_rank_constant: int
    rank_window_size: int
    sparse_weight: float
    vector_weight: float
    sparse_base_url: str
    vector_base_url: str
    http_timeout_seconds: float
    auto_rebuild_on_search: bool


@lru_cache
def get_hybrid_retrieval_settings() -> HybridRetrievalSettings:
    return HybridRetrievalSettings()


def get_hybrid_retrieval_config(
    settings: HybridRetrievalSettings | None = None,
) -> HybridRetrievalRuntimeConfig:
    cfg = settings or get_hybrid_retrieval_settings()
    return HybridRetrievalRuntimeConfig(
        top_k_default=cfg.top_k_default,
        top_k_max=cfg.top_k_max,
        rrf_rank_constant=cfg.rrf_rank_constant,
        rank_window_size=cfg.rank_window_size,
        sparse_weight=cfg.sparse_weight,
        vector_weight=cfg.vector_weight,
        sparse_base_url=cfg.sparse_base_url.rstrip("/"),
        vector_base_url=cfg.vector_base_url.rstrip("/"),
        http_timeout_seconds=cfg.http_timeout_seconds,
        auto_rebuild_on_search=cfg.auto_rebuild_on_search,
    )
