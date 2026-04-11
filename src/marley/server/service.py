"""Pipeline service for the MARley server.

Manages retriever instances with lazy loading and caching,
and orchestrates the full pipeline (retrieval -> abstention -> generation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.marley.server.pipeline import run_with_abstention
from src.marley.generator.ollama import OllamaGenerator
from src.marley.models.constants import (
    DEFAULT_K_RRF,
    DEFAULT_K_RRF_HYBRID,
    DEFAULT_THRESHOLDS,
    NORMALIZATION_MAP,
    SOURCE_TEXT_TRUNCATION,
)
from src.marley.retrieval import (
    BM25Retriever,
    FusionRetriever,
    HybridRetriever,
    MergedRetriever,
    Retriever,
    VectorRetriever,
    load_chunks,
)
from src.marley.server.config import CHUNK_PATHS, ServerConfig

logger = logging.getLogger(__name__)


class PipelineService:
    """Manages the MARley pipeline with lazy retriever caching.

    Retrievers are created and indexed on first use, then cached
    for the server lifetime. The cache key is a tuple of
    (retriever_type, frozenset(knowledge_bases), strategy).
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._generator = OllamaGenerator(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        )
        self._retriever_cache: dict[tuple, Retriever] = {}

    @property
    def generator_model(self) -> str:
        """Return the configured LLM model name."""
        return self._config.ollama_model

    @property
    def cached_retriever_count(self) -> int:
        """Return the number of cached retriever instances."""
        return len(self._retriever_cache)

    def available_knowledge_bases(self) -> list[str]:
        """Return knowledge bases for which chunk files exist."""
        available = []
        for kb, rel_path in CHUNK_PATHS.items():
            if Path(rel_path).exists():
                available.append(kb)
        return available

    def _vector_persist_dir(self, tag: str) -> str:
        """Return a unique ChromaDB persist directory for a given tag."""
        return str(self._config.chunk_dir / ".chromadb" / tag)

    def _create_retriever(self, retriever_type: str, tag: str = "default", k_rrf_hybrid: int = DEFAULT_K_RRF_HYBRID) -> Retriever:
        """Create a new (unindexed) retriever instance.

        Args:
            retriever_type: "bm25", "vector", or "hybrid".
            tag: Unique identifier for the ChromaDB persist directory
                 (used by vector and hybrid retrievers).
        """
        if retriever_type == "bm25":
            return BM25Retriever()
        if retriever_type == "vector":
            return VectorRetriever(persist_directory=self._vector_persist_dir(f"vector-{tag}"))
        if retriever_type == "hybrid":
            return HybridRetriever((
                BM25Retriever(),
                VectorRetriever(persist_directory=self._vector_persist_dir(f"hybrid-{tag}")),
            ), k_rrf=k_rrf_hybrid)
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    def _load_and_merge_chunks(self, knowledge_bases: list[str]) -> list[dict[str, Any]]:
        """Load and merge chunks from the specified knowledge bases."""
        all_chunks: list[dict[str, Any]] = []
        for kb in knowledge_bases:
            path = CHUNK_PATHS.get(kb)
            if not path:
                raise ValueError(f"Unknown knowledge base: {kb}")
            chunks = load_chunks(path)
            all_chunks.extend(chunks)
        return all_chunks

    def get_retriever(
        self,
        retriever_type: str,
        knowledge_bases: list[str],
        strategy: str,
        k_rrf: int = DEFAULT_K_RRF,
    ) -> Retriever:
        """Get or create a cached retriever for the given configuration.

        Args:
            retriever_type: "bm25", "vector", or "hybrid".
            knowledge_bases: List of KB identifiers.
            strategy: "single", "merged_pool", or "fusion".
            k_rrf: RRF smoothing constant (for fusion strategy).

        Returns:
            An indexed Retriever instance.
        """
        cache_key = (retriever_type, frozenset(knowledge_bases), strategy)

        if cache_key in self._retriever_cache:
            return self._retriever_cache[cache_key]

        logger.info(
            "Building retriever: type=%s, kbs=%s, strategy=%s",
            retriever_type, knowledge_bases, strategy,
        )

        # Build a unique tag for ChromaDB persist directories
        kb_tag = "-".join(sorted(knowledge_bases))

        if strategy == "fusion":
            sub_retrievers: list[Retriever] = []
            for kb in knowledge_bases:
                r = self._create_retriever(retriever_type, tag=f"fusion-{kb}")
                chunks = load_chunks(CHUNK_PATHS[kb])
                r.index(chunks)
                sub_retrievers.append(r)
            retriever = FusionRetriever(sub_retrievers, k_rrf=k_rrf)
        else:
            inner = self._create_retriever(retriever_type, tag=f"{strategy}-{kb_tag}")
            retriever = MergedRetriever(inner)
            chunks = self._load_and_merge_chunks(knowledge_bases)
            retriever.index(chunks)

        self._retriever_cache[cache_key] = retriever
        logger.info("Retriever cached: %s (%d chunks)", cache_key, retriever.size)
        return retriever

    def chat(
        self,
        query: str,
        *,
        retriever_type: str = "hybrid",
        knowledge_bases: list[str] | None = None,
        strategy: str = "merged_pool",
        k: int = 5,
        threshold: float | None = None,
        k_rrf: int = DEFAULT_K_RRF,
    ) -> dict[str, Any]:
        """Run the full pipeline and return a structured result.

        Args:
            query: The user question.
            retriever_type: Retriever type to use.
            knowledge_bases: KBs to search (default: all).
            strategy: Combination strategy.
            k: Number of chunks to retrieve.
            threshold: Abstention threshold (None = use default for type).
            k_rrf: RRF smoothing constant.

        Returns:
            Dict with answer, sources, confidence, abstention info,
            and configuration metadata.
        """
        kbs = knowledge_bases or self._config.default_knowledge_bases

        # Validate retriever type
        if retriever_type not in NORMALIZATION_MAP:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        # Determine normalization strategy
        if strategy == "fusion":
            norm_strategy = "rrf"
        else:
            norm_strategy = NORMALIZATION_MAP[retriever_type]

        # Determine threshold
        if threshold is None:
            threshold = DEFAULT_THRESHOLDS.get(norm_strategy, 0.3)

        # Get or create retriever
        retriever = self.get_retriever(retriever_type, kbs, strategy, k_rrf)

        # Determine normalization params
        norm_params: dict[str, Any] = {}
        if norm_strategy == "rrf":
            n_retrievers = len(kbs) if strategy == "fusion" else 2
            norm_params = {"rrf_n_retrievers": n_retrievers, "rrf_k": k_rrf}

        # Run pipeline
        result = run_with_abstention(
            query,
            retriever,
            self._generator,
            k=k,
            threshold=threshold,
            normalization_strategy=norm_strategy,
            normalization_params=norm_params,
        )

        # Build source references (include page metadata for PDF viewer)
        sources = [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"][:SOURCE_TEXT_TRUNCATION],
                "score": round(r["score"], 4),
                "metadata": {
                    "start_page": r.get("metadata", {}).get("start_page"),
                    "end_page": r.get("metadata", {}).get("end_page"),
                    "section_title": r.get("metadata", {}).get("section_title"),
                },
            }
            for r in result.retrieval_results
        ]

        return {
            "answer": result.answer,
            "abstained": result.abstained,
            "abstention_level": result.level,
            "abstention_reason": result.reason,
            "confidence": round(result.confidence, 4),
            "sources": sources,
            "config": {
                "retriever_type": retriever_type,
                "knowledge_bases": kbs,
                "strategy": strategy,
                "k": k,
                "threshold": threshold,
                "normalization_strategy": norm_strategy,
                "model": result.model or self._config.ollama_model,
            },
        }
