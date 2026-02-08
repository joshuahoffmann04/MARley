from __future__ import annotations

import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generator.config import GeneratorSettings, get_generator_config
from generator.models import (
    GenerateRequest,
    RetrievalHit,
    RetrievalQualityFlag,
    RetrievalSearchResponse,
    SourceType,
)
from generator.service import GeneratorService
from MARley.config import MarleyRuntimeConfig, MarleySettings, get_marley_config, get_marley_settings
from MARley.models import (
    ChatRequest,
    ChatResponse,
    DocumentOption,
    DocumentSourceAvailability,
    OptionsResponse,
    RetrievalMode,
)
from retrieval.sparse_retrieval.config import SparseRetrievalSettings, get_sparse_retrieval_config
from retrieval.sparse_retrieval.service import SparseBM25Retriever
from retrieval.vector_retrieval.config import VectorRetrievalSettings, get_vector_retrieval_config
from retrieval.vector_retrieval.service import VectorRetriever

SOURCE_GLOBS: dict[SourceType, str] = {
    "pdf": "*-pdf-chunker-*.json",
    "faq_so": "*-faq-chunker-so-*.json",
    "faq_sb": "*-faq-chunker-sb-*.json",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_flag(value: Any) -> RetrievalQualityFlag | None:
    payload: dict[str, Any]
    if hasattr(value, "model_dump"):
        payload = value.model_dump()
    elif isinstance(value, dict):
        payload = value
    else:
        return None

    code = str(payload.get("code") or "UNKNOWN_FLAG")
    message = str(payload.get("message") or "No message.")
    severity = str(payload.get("severity") or "warning")
    if severity not in {"info", "warning", "error"}:
        severity = "warning"
    context = payload.get("context")
    if not isinstance(context, dict):
        context = {}
    return RetrievalQualityFlag(
        code=code,
        message=message,
        severity=severity,  # type: ignore[arg-type]
        context=context,
    )


class MarleyPipelineService:
    def __init__(
        self,
        *,
        settings: MarleySettings | None = None,
        runtime_config: MarleyRuntimeConfig | None = None,
    ) -> None:
        self.settings = settings or get_marley_settings()
        self.runtime_config = runtime_config or get_marley_config(self.settings)

        sparse_settings = SparseRetrievalSettings(
            document_id=self.runtime_config.default_document_id,
            data_root=self.runtime_config.data_root,
        )
        self.sparse_retriever = SparseBM25Retriever(
            settings=sparse_settings,
            runtime_config=get_sparse_retrieval_config(sparse_settings),
        )

        vector_settings = VectorRetrievalSettings(
            document_id=self.runtime_config.default_document_id,
            data_root=self.runtime_config.data_root,
            chroma_client_mode="persistent",
            chroma_persist_subdir=self.runtime_config.vector_persist_subdir,
            ollama_embedding_url=f"{self.runtime_config.ollama_base_url}/api/embeddings",
            ollama_embedding_model=self.runtime_config.embedding_model,
        )
        self.vector_retriever = VectorRetriever(
            settings=vector_settings,
            runtime_config=get_vector_retrieval_config(vector_settings),
        )

        generator_settings = GeneratorSettings(
            document_id=self.runtime_config.default_document_id,
            ollama_base_url=self.runtime_config.ollama_base_url,
            ollama_model=self.runtime_config.generator_model,
            temperature_default=self.runtime_config.generator_temperature,
            total_budget_tokens_default=self.runtime_config.generator_total_budget_tokens,
            max_answer_tokens_default=self.runtime_config.generator_max_answer_tokens,
        )
        self.generator = GeneratorService(
            settings=generator_settings,
            runtime_config=get_generator_config(generator_settings),
        )

    def _chunks_dir_for_document(self, document_id: str) -> Path:
        return (self.runtime_config.data_root / document_id / "chunks").resolve()

    def _latest_source_file(self, *, document_id: str, source_type: SourceType) -> Path | None:
        chunks_dir = self._chunks_dir_for_document(document_id)
        if not chunks_dir.exists():
            return None

        source_glob = SOURCE_GLOBS[source_type]
        all_candidates = sorted(
            chunks_dir.glob(source_glob),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not all_candidates:
            return None

        doc_id_lc = document_id.lower()
        doc_candidates = [path for path in all_candidates if doc_id_lc in path.name.lower()]
        if not doc_candidates:
            return None
        return doc_candidates[0].resolve()

    def _available_sources(self, document_id: str) -> dict[SourceType, Path | None]:
        return {
            "pdf": self._latest_source_file(document_id=document_id, source_type="pdf"),
            "faq_so": self._latest_source_file(document_id=document_id, source_type="faq_so"),
            "faq_sb": self._latest_source_file(document_id=document_id, source_type="faq_sb"),
        }

    def _document_ids(self) -> list[str]:
        if not self.runtime_config.data_root.exists():
            return []

        document_ids: list[str] = []
        for candidate in sorted(self.runtime_config.data_root.iterdir(), key=lambda item: item.name.lower()):
            if not candidate.is_dir():
                continue
            document_id = candidate.name
            raw_dir = candidate / "raw"
            chunks_dir = candidate / "chunks"
            if raw_dir.exists() or chunks_dir.exists():
                document_ids.append(document_id)
        return document_ids

    def options(self) -> OptionsResponse:
        documents: list[DocumentOption] = []
        for document_id in self._document_ids():
            available = self._available_sources(document_id)
            sources: list[DocumentSourceAvailability] = []
            for source_type in ("pdf", "faq_so", "faq_sb"):
                source_path = available[source_type]
                sources.append(
                    DocumentSourceAvailability(
                        source_type=source_type,  # type: ignore[arg-type]
                        available=source_path is not None,
                        latest_file=str(source_path) if source_path else None,
                        last_modified=(
                            datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
                            if source_path
                            else None
                        ),
                    )
                )
            documents.append(DocumentOption(document_id=document_id, sources=sources))

        default_document_id: str | None = None
        available_ids = {item.document_id for item in documents}
        if self.runtime_config.default_document_id in available_ids:
            default_document_id = self.runtime_config.default_document_id
        elif documents:
            default_document_id = documents[0].document_id

        return OptionsResponse(
            documents=documents,
            default_document_id=default_document_id,
            default_retrieval_mode=self.runtime_config.default_retrieval_mode,
        )

    def _resolve_source_types(
        self,
        *,
        document_id: str,
        requested: list[SourceType] | None,
    ) -> list[SourceType]:
        available = self._available_sources(document_id)
        if requested:
            filtered = [source_type for source_type in requested if available.get(source_type) is not None]
        else:
            filtered = [source_type for source_type, path in available.items() if path is not None]

        if not filtered:
            raise ValueError("No selectable sources available for this document.")
        return filtered

    def _convert_sparse_to_common(self, *, response: Any, top_k: int) -> RetrievalSearchResponse:
        flags: list[RetrievalQualityFlag] = []
        for raw_flag in response.quality_flags:
            parsed = _to_flag(raw_flag)
            if parsed:
                flags.append(parsed)

        hits: list[RetrievalHit] = []
        for hit in response.hits:
            hits.append(
                RetrievalHit(
                    rank=hit.rank,
                    rrf_score=1.0 / (self.runtime_config.rrf_rank_constant + hit.rank),
                    source_type=hit.source_type,
                    chunk_id=hit.chunk_id,
                    chunk_type=hit.chunk_type,
                    text=hit.text,
                    token_count=hit.token_count,
                    metadata=hit.metadata,
                    input_file=hit.input_file,
                    sparse_rank=hit.rank,
                    sparse_score=hit.score,
                    vector_rank=None,
                    vector_score=None,
                    vector_distance=None,
                )
            )

        return RetrievalSearchResponse(
            document_id=response.document_id,
            query=response.query,
            top_k=top_k,
            hits=hits[:top_k],
            quality_flags=flags,
        )

    def _convert_vector_to_common(self, *, response: Any, top_k: int) -> RetrievalSearchResponse:
        flags: list[RetrievalQualityFlag] = []
        for raw_flag in response.quality_flags:
            parsed = _to_flag(raw_flag)
            if parsed:
                flags.append(parsed)

        hits: list[RetrievalHit] = []
        for hit in response.hits:
            score_value = hit.score if hit.score is not None else 0.0
            hits.append(
                RetrievalHit(
                    rank=hit.rank,
                    rrf_score=1.0 / (self.runtime_config.rrf_rank_constant + hit.rank),
                    source_type=hit.source_type,
                    chunk_id=hit.chunk_id,
                    chunk_type=hit.chunk_type,
                    text=hit.text,
                    token_count=hit.token_count,
                    metadata=hit.metadata,
                    input_file=hit.input_file,
                    sparse_rank=None,
                    sparse_score=None,
                    vector_rank=hit.rank,
                    vector_score=score_value,
                    vector_distance=hit.distance,
                )
            )

        return RetrievalSearchResponse(
            document_id=response.document_id,
            query=response.query,
            top_k=top_k,
            hits=hits[:top_k],
            quality_flags=flags,
        )

    def _backend_failure_flag(self, *, backend: str, error: Exception) -> RetrievalQualityFlag:
        return RetrievalQualityFlag(
            code="BACKEND_FAILED",
            message="Retrieval backend failed during search.",
            severity="warning",
            context={"backend": backend, "error": str(error)},
        )

    def _fuse_hybrid(
        self,
        *,
        query: str,
        document_id: str,
        top_k: int,
        sparse_hits: list[RetrievalHit],
        vector_hits: list[RetrievalHit],
        flags: list[RetrievalQualityFlag],
    ) -> RetrievalSearchResponse:
        fused: dict[tuple[SourceType, str], RetrievalHit] = {}

        def ingest(hits: list[RetrievalHit], *, backend: str, weight: float) -> None:
            for hit in hits:
                key = (hit.source_type, hit.chunk_id)
                candidate = fused.get(key)
                contrib = weight * (1.0 / (self.runtime_config.rrf_rank_constant + hit.rank))
                if candidate is None:
                    candidate = RetrievalHit(
                        rank=0,
                        rrf_score=0.0,
                        source_type=hit.source_type,
                        chunk_id=hit.chunk_id,
                        chunk_type=hit.chunk_type,
                        text=hit.text,
                        token_count=hit.token_count,
                        metadata=hit.metadata,
                        input_file=hit.input_file,
                        sparse_rank=hit.sparse_rank,
                        sparse_score=hit.sparse_score,
                        vector_rank=hit.vector_rank,
                        vector_score=hit.vector_score,
                        vector_distance=hit.vector_distance,
                    )
                    fused[key] = candidate
                else:
                    if candidate.sparse_rank is None and hit.sparse_rank is not None:
                        candidate.sparse_rank = hit.sparse_rank
                        candidate.sparse_score = hit.sparse_score
                    if candidate.vector_rank is None and hit.vector_rank is not None:
                        candidate.vector_rank = hit.vector_rank
                        candidate.vector_score = hit.vector_score
                        candidate.vector_distance = hit.vector_distance

                candidate.rrf_score += contrib
                if backend == "sparse" and candidate.sparse_rank is None:
                    candidate.sparse_rank = hit.rank
                    candidate.sparse_score = hit.sparse_score
                if backend == "vector" and candidate.vector_rank is None:
                    candidate.vector_rank = hit.rank
                    candidate.vector_score = hit.vector_score
                    candidate.vector_distance = hit.vector_distance

        ingest(
            sparse_hits,
            backend="sparse",
            weight=self.runtime_config.sparse_weight,
        )
        ingest(
            vector_hits,
            backend="vector",
            weight=self.runtime_config.vector_weight,
        )

        sorted_hits = sorted(
            fused.values(),
            key=lambda item: (
                -item.rrf_score,
                min([rank for rank in (item.sparse_rank, item.vector_rank) if rank is not None] or [10**9]),
                item.source_type,
                item.chunk_id,
            ),
        )

        ranked: list[RetrievalHit] = []
        for index, hit in enumerate(sorted_hits[:top_k], start=1):
            hit.rank = index
            ranked.append(hit)

        if not ranked:
            flags.append(
                RetrievalQualityFlag(
                    code="HYBRID_NO_HITS",
                    message="Hybrid fusion produced no hits.",
                    severity="warning",
                )
            )

        return RetrievalSearchResponse(
            document_id=document_id,
            query=query,
            top_k=top_k,
            hits=ranked,
            quality_flags=flags,
        )

    def search(
        self,
        *,
        query: str,
        document_id: str,
        retrieval_mode: RetrievalMode,
        source_types: list[SourceType],
    ) -> RetrievalSearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        if retrieval_mode == "sparse":
            response = self.sparse_retriever.search(
                query=normalized_query,
                document_id=document_id,
                top_k=self.runtime_config.top_k_default,
                source_types=source_types,
                rebuild_if_stale=True,
            )
            return self._convert_sparse_to_common(response=response, top_k=self.runtime_config.top_k_default)

        if retrieval_mode == "vector":
            response = self.vector_retriever.search(
                query=normalized_query,
                document_id=document_id,
                top_k=self.runtime_config.top_k_default,
                source_types=source_types,
                rebuild_if_stale=True,
            )
            return self._convert_vector_to_common(response=response, top_k=self.runtime_config.top_k_default)

        backend_top_k = max(self.runtime_config.top_k_default, self.runtime_config.rank_window_size)
        sparse_common: RetrievalSearchResponse | None = None
        vector_common: RetrievalSearchResponse | None = None
        flags: list[RetrievalQualityFlag] = []

        try:
            sparse_response = self.sparse_retriever.search(
                query=normalized_query,
                document_id=document_id,
                top_k=backend_top_k,
                source_types=source_types,
                rebuild_if_stale=True,
            )
            sparse_common = self._convert_sparse_to_common(response=sparse_response, top_k=backend_top_k)
            flags.extend(sparse_common.quality_flags)
        except Exception as exc:
            flags.append(self._backend_failure_flag(backend="sparse", error=exc))

        try:
            vector_response = self.vector_retriever.search(
                query=normalized_query,
                document_id=document_id,
                top_k=backend_top_k,
                source_types=source_types,
                rebuild_if_stale=True,
            )
            vector_common = self._convert_vector_to_common(response=vector_response, top_k=backend_top_k)
            flags.extend(vector_common.quality_flags)
        except Exception as exc:
            flags.append(self._backend_failure_flag(backend="vector", error=exc))

        if sparse_common is None and vector_common is None:
            raise ValueError("Neither sparse nor vector retrieval is currently available.")

        sparse_hits = sparse_common.hits if sparse_common else []
        vector_hits = vector_common.hits if vector_common else []
        return self._fuse_hybrid(
            query=normalized_query,
            document_id=document_id,
            top_k=self.runtime_config.top_k_default,
            sparse_hits=sparse_hits,
            vector_hits=vector_hits,
            flags=flags,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        source_types = self._resolve_source_types(
            document_id=request.document_id,
            requested=request.source_types,
        )

        retrieval_response = self.search(
            query=request.query,
            document_id=request.document_id,
            retrieval_mode=request.retrieval_mode,
            source_types=source_types,
        )

        generator_request = GenerateRequest(
            query=request.query,
            document_id=request.document_id,
            top_k=self.runtime_config.top_k_default,
            source_types=source_types,
            model=self.runtime_config.generator_model,
            temperature=self.runtime_config.generator_temperature,
            total_budget_tokens=self.runtime_config.generator_total_budget_tokens,
            max_answer_tokens=self.runtime_config.generator_max_answer_tokens,
            include_used_chunks=True,
        )
        generator_response = self.generator.generate_from_retrieval(
            request=generator_request,
            retrieval=retrieval_response,
        )

        return ChatResponse(
            answer=generator_response.answer,
            abstained=generator_response.abstained,
            document_id=request.document_id,
            retrieval_mode=request.retrieval_mode,
            source_types=source_types,
            generated_at=generator_response.generated_at,
            generator_response=generator_response.model_dump(mode="json"),
            retrieval_response=retrieval_response.model_dump(mode="json"),
        )

    def readiness(self) -> dict[str, Any]:
        sparse_status = self.sparse_retriever.readiness()
        vector_status = self.vector_retriever.readiness()

        ollama_status: dict[str, Any] = {"available": False}
        request_obj = urllib.request.Request(
            url=f"{self.runtime_config.ollama_base_url}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request_obj, timeout=10.0) as response:
                ollama_status = {
                    "available": 200 <= int(response.status) < 300,
                    "status_code": int(response.status),
                }
        except urllib.error.HTTPError as exc:
            ollama_status = {
                "available": False,
                "status_code": int(exc.code),
                "error": str(exc),
            }
        except urllib.error.URLError as exc:
            ollama_status = {"available": False, "error": str(exc.reason)}
        except socket.timeout:
            ollama_status = {"available": False, "error": "request timed out"}

        overall = (
            "ready"
            if ollama_status.get("available") and sparse_status.get("status") == "ready"
            else "degraded"
        )
        return {
            "status": overall,
            "timestamp": _now_utc().isoformat(),
            "dependencies": {
                "ollama": ollama_status,
                "sparse": sparse_status,
                "vector": vector_status,
            },
        }
