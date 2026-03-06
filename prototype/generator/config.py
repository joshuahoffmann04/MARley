from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneratorSettings(BaseSettings):
    app_name: str = "MARley"
    environment: str = "development"

    document_id: str = "msc-computer-science"

    retrieval_base_url: str = "http://127.0.0.1:8006"
    retrieval_timeout_seconds: float = Field(default=30.0, gt=0.0)
    retrieval_rebuild_if_stale_default: bool = True

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:latest"
    ollama_timeout_seconds: float = Field(default=120.0, gt=0.0)

    temperature_default: float = Field(default=0.1, ge=0.0, le=2.0)

    top_k_default: int = Field(default=10, ge=1)
    top_k_max: int = Field(default=100, ge=1)

    total_budget_tokens_default: int = Field(default=2048, ge=256)
    total_budget_tokens_max: int = Field(default=32768, ge=256)
    max_answer_tokens_default: int = Field(default=384, ge=16)
    max_answer_tokens_max: int = Field(default=8192, ge=16)
    prompt_overhead_tokens: int = Field(default=320, ge=64)

    min_hits_default: int = Field(default=2, ge=1)
    min_best_rrf_score_default: float = Field(default=0.015, ge=0.0)
    min_dual_backend_hits_default: int = Field(default=1, ge=0)
    abstain_on_retrieval_errors_default: bool = True
    abstain_on_backend_degradation_default: bool = True
    abstain_on_low_confidence_default: bool = True
    abstention_answer_text: str = "Ich weiß es nicht."

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GENERATOR_",
        extra="ignore",
    )


@dataclass(frozen=True)
class GeneratorRuntimeConfig:
    document_id: str

    retrieval_base_url: str
    retrieval_timeout_seconds: float
    retrieval_rebuild_if_stale_default: bool

    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: float

    temperature_default: float

    top_k_default: int
    top_k_max: int

    total_budget_tokens_default: int
    total_budget_tokens_max: int
    max_answer_tokens_default: int
    max_answer_tokens_max: int
    prompt_overhead_tokens: int

    min_hits_default: int
    min_best_rrf_score_default: float
    min_dual_backend_hits_default: int
    abstain_on_retrieval_errors_default: bool
    abstain_on_backend_degradation_default: bool
    abstain_on_low_confidence_default: bool
    abstention_answer_text: str


@lru_cache
def get_generator_settings() -> GeneratorSettings:
    return GeneratorSettings()


def get_generator_config(settings: GeneratorSettings | None = None) -> GeneratorRuntimeConfig:
    cfg = settings or get_generator_settings()

    return GeneratorRuntimeConfig(
        document_id=cfg.document_id,
        retrieval_base_url=cfg.retrieval_base_url.rstrip("/"),
        retrieval_timeout_seconds=cfg.retrieval_timeout_seconds,
        retrieval_rebuild_if_stale_default=cfg.retrieval_rebuild_if_stale_default,
        ollama_base_url=cfg.ollama_base_url.rstrip("/"),
        ollama_model=cfg.ollama_model,
        ollama_timeout_seconds=cfg.ollama_timeout_seconds,
        temperature_default=cfg.temperature_default,
        top_k_default=cfg.top_k_default,
        top_k_max=cfg.top_k_max,
        total_budget_tokens_default=cfg.total_budget_tokens_default,
        total_budget_tokens_max=cfg.total_budget_tokens_max,
        max_answer_tokens_default=cfg.max_answer_tokens_default,
        max_answer_tokens_max=cfg.max_answer_tokens_max,
        prompt_overhead_tokens=cfg.prompt_overhead_tokens,
        min_hits_default=cfg.min_hits_default,
        min_best_rrf_score_default=cfg.min_best_rrf_score_default,
        min_dual_backend_hits_default=cfg.min_dual_backend_hits_default,
        abstain_on_retrieval_errors_default=cfg.abstain_on_retrieval_errors_default,
        abstain_on_backend_degradation_default=cfg.abstain_on_backend_degradation_default,
        abstain_on_low_confidence_default=cfg.abstain_on_low_confidence_default,
        abstention_answer_text=cfg.abstention_answer_text,
    )
