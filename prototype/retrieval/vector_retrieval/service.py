from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from retrieval.vector_retrieval.config import (
    VectorRetrievalRuntimeConfig,
    VectorRetrievalSettings,
    get_vector_retrieval_config,
    get_vector_retrieval_settings,
)
from retrieval.vector_retrieval.models import (
    IndexStats,
    RetrievalQualityFlag,
    SearchResponse,
    SourceSnapshot,
    SourceType,
    VectorSearchHit,
)

SOURCE_DEFINITIONS: tuple[tuple[SourceType, str], ...] = (
    ("pdf", "pdf_chunks_glob"),
    ("faq_so", "faq_so_chunks_glob"),
    ("faq_sb", "faq_sb_chunks_glob"),
)


def _patch_posthog_capture_if_incompatible() -> None:
    """Patch newer posthog capture signature to avoid chroma telemetry crashes.

    Chroma 0.6.x still calls posthog.capture(distinct_id, event, properties),
    while posthog 7.x expects capture(event, **kwargs). We disable capture to
    keep runtime stable and silent.
    """
    try:
        import posthog  # type: ignore
    except Exception:
        return

    if getattr(posthog, "_marley_capture_patched", False):
        return

    capture_func = getattr(posthog, "capture", None)
    if not callable(capture_func):
        return

    try:
        signature = inspect.signature(capture_func)
    except Exception:
        return

    params = list(signature.parameters.values())
    # posthog 7.x: capture(event, **kwargs)
    if len(params) == 2 and params[0].name == "event":
        def _capture_noop(*args: Any, **kwargs: Any) -> None:
            return None

        posthog.capture = _capture_noop
        posthog.disabled = True
        setattr(posthog, "_marley_capture_patched", True)


class VectorBackendUnavailableError(RuntimeError):
    """Raised when the vector backend (Chroma/Ollama) is not reachable."""


@dataclass(frozen=True)
class _IndexedDocument:
    vector_id: str
    source_type: SourceType
    input_file: str
    chunk_id: str
    chunk_type: str
    text: str
    token_count: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _IndexState:
    document_id: str
    index_signature: str
    collection_name: str
    built_at: datetime
    documents_by_vector_id: dict[str, _IndexedDocument]
    index_stats: IndexStats
    quality_flags: list[RetrievalQualityFlag]


class VectorRetriever:
    def __init__(
        self,
        *,
        settings: VectorRetrievalSettings | None = None,
        runtime_config: VectorRetrievalRuntimeConfig | None = None,
    ) -> None:
        _patch_posthog_capture_if_incompatible()

        self.settings = settings or get_vector_retrieval_settings()
        self.runtime_config = runtime_config or get_vector_retrieval_config(self.settings)
        self._lock = Lock()
        self._state: _IndexState | None = None
        self._http_client: ClientAPI | None = None
        self._memory_client: ClientAPI | None = None
        self._persistent_clients: dict[str, ClientAPI] = {}
        self._persistent_fallback_reason: str | None = None
        self._chroma_settings = Settings(anonymized_telemetry=False)
        self._embedding_function = OllamaEmbeddingFunction(
            url=self.runtime_config.ollama_embedding_url,
            model_name=self.runtime_config.ollama_embedding_model,
        )

    def _input_dir_for_document(self, *, document_id: str) -> Path:
        if self.settings.input_dir is not None:
            return self.runtime_config.input_dir
        return (self.settings.data_root_path / document_id / "chunks").resolve()

    def _persistent_chroma_path_for_document(self, *, document_id: str) -> Path:
        if self.runtime_config.chroma_persist_dir is not None:
            return self.runtime_config.chroma_persist_dir
        return (self.settings.data_root_path / document_id / self.runtime_config.chroma_persist_subdir).resolve()

    def _get_memory_client(self) -> ClientAPI:
        if self._memory_client is None:
            self._memory_client = chromadb.Client()
        return self._memory_client

    def _get_client(self, *, document_id: str) -> ClientAPI:
        if self.runtime_config.chroma_client_mode == "persistent":
            if self._persistent_fallback_reason is not None:
                return self._get_memory_client()

            persist_path = self._persistent_chroma_path_for_document(document_id=document_id)
            persist_path.mkdir(parents=True, exist_ok=True)
            cache_key = str(persist_path)
            client = self._persistent_clients.get(cache_key)
            if client is None:
                try:
                    client = chromadb.PersistentClient(
                        path=str(persist_path),
                        settings=self._chroma_settings,
                        tenant=self.runtime_config.chroma_tenant,
                        database=self.runtime_config.chroma_database,
                    )
                    self._persistent_clients[cache_key] = client
                except Exception as exc:
                    self._persistent_fallback_reason = (
                        "Persistent Chroma backend could not be initialized; "
                        "falling back to in-memory mode. "
                        f"Reason: {exc}"
                    )
                    return self._get_memory_client()
            return client

        if self._http_client is None:
            self._http_client = chromadb.HttpClient(
                host=self.runtime_config.chroma_host,
                port=self.runtime_config.chroma_port,
                ssl=self.runtime_config.chroma_ssl,
                settings=self._chroma_settings,
                tenant=self.runtime_config.chroma_tenant,
                database=self.runtime_config.chroma_database,
            )
        return self._http_client

    def _check_backend_health(self, *, document_id: str) -> None:
        try:
            client = self._get_client(document_id=document_id)
            client.heartbeat()
        except Exception as exc:  # pragma: no cover - external service dependent
            if self.runtime_config.chroma_client_mode == "http":
                raise VectorBackendUnavailableError(
                    "Chroma HTTP backend is not reachable. "
                    f"Configured endpoint: {self.runtime_config.chroma_host}:{self.runtime_config.chroma_port}"
                ) from exc
            self._persistent_fallback_reason = (
                "Persistent Chroma backend is not reachable; switched to in-memory mode. "
                f"Reason: {exc}"
            )
            try:
                memory_client = self._get_memory_client()
                memory_client.heartbeat()
                return
            except Exception as memory_exc:
                raise VectorBackendUnavailableError(
                    "Chroma persistent backend is not reachable."
                ) from memory_exc

    def _collection_name(self, document_id: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", document_id).strip("-").lower()
        if not normalized:
            normalized = "default"
        return f"{self.runtime_config.chroma_collection_prefix}-{normalized}"

    def _resolve_latest_source_file(
        self,
        *,
        input_dir: Path,
        document_id: str,
        source_glob: str,
    ) -> Path | None:
        all_candidates = sorted(
            input_dir.glob(source_glob),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not all_candidates:
            return None

        doc_id_lc = document_id.lower()
        document_candidates = [path for path in all_candidates if doc_id_lc in path.name.lower()]
        if not document_candidates:
            return None
        return document_candidates[0].resolve()

    def _collect_source_files(
        self,
        *,
        document_id: str,
    ) -> tuple[dict[SourceType, Path | None], list[SourceSnapshot], list[RetrievalQualityFlag]]:
        input_dir = self._input_dir_for_document(document_id=document_id)
        source_files: dict[SourceType, Path | None] = {}
        snapshots: list[SourceSnapshot] = []
        flags: list[RetrievalQualityFlag] = []

        for source_type, glob_attr in SOURCE_DEFINITIONS:
            source_glob = getattr(self.runtime_config, glob_attr)
            source_file = self._resolve_latest_source_file(
                input_dir=input_dir,
                document_id=document_id,
                source_glob=source_glob,
            )
            source_files[source_type] = source_file

            if source_file is None:
                snapshots.append(
                    SourceSnapshot(
                        source_type=source_type,
                        missing=True,
                    )
                )
                flags.append(
                    RetrievalQualityFlag(
                        code="SOURCE_FILE_MISSING",
                        message="Expected vector source file was not found.",
                        severity="info",
                        context={
                            "document_id": document_id,
                            "source_type": source_type,
                            "glob": source_glob,
                            "input_dir": str(input_dir),
                        },
                    )
                )
            else:
                snapshots.append(
                    SourceSnapshot(
                        source_type=source_type,
                        file_path=str(source_file),
                        missing=False,
                        last_modified=datetime.fromtimestamp(
                            source_file.stat().st_mtime, tz=timezone.utc
                        ),
                    )
                )

        return source_files, snapshots, flags

    def _load_documents_from_source(
        self,
        *,
        source_type: SourceType,
        source_file: Path,
    ) -> tuple[list[_IndexedDocument], list[RetrievalQualityFlag]]:
        flags: list[RetrievalQualityFlag] = []
        documents: list[_IndexedDocument] = []

        try:
            payload = json.loads(source_file.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            flags.append(
                RetrievalQualityFlag(
                    code="SOURCE_FILE_READ_ERROR",
                    message="Source file could not be parsed as JSON.",
                    severity="error",
                    context={"source_type": source_type, "file_path": str(source_file), "error": str(exc)},
                )
            )
            return documents, flags

        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            flags.append(
                RetrievalQualityFlag(
                    code="SOURCE_CHUNKS_INVALID",
                    message="Source JSON does not contain a valid 'chunks' list.",
                    severity="error",
                    context={"source_type": source_type, "file_path": str(source_file)},
                )
            )
            return documents, flags

        for index, raw_chunk in enumerate(raw_chunks):
            if not isinstance(raw_chunk, dict):
                flags.append(
                    RetrievalQualityFlag(
                        code="CHUNK_RECORD_INVALID",
                        message="Chunk record is not an object and was skipped.",
                        severity="warning",
                        context={
                            "source_type": source_type,
                            "file_path": str(source_file),
                            "chunk_index": index,
                        },
                    )
                )
                continue

            chunk_id = str(raw_chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                flags.append(
                    RetrievalQualityFlag(
                        code="CHUNK_ID_MISSING",
                        message="Chunk record has no chunk_id and was skipped.",
                        severity="warning",
                        context={
                            "source_type": source_type,
                            "file_path": str(source_file),
                            "chunk_index": index,
                        },
                    )
                )
                continue

            text = str(raw_chunk.get("text") or "").strip()
            if not text:
                flags.append(
                    RetrievalQualityFlag(
                        code="CHUNK_TEXT_EMPTY",
                        message="Chunk has empty text and was skipped.",
                        severity="info",
                        context={"source_type": source_type, "chunk_id": chunk_id},
                    )
                )
                continue

            token_count_raw = raw_chunk.get("token_count")
            token_count = token_count_raw if isinstance(token_count_raw, int) else None
            metadata_obj = raw_chunk.get("metadata")
            if not isinstance(metadata_obj, dict):
                metadata_obj = {}

            vector_id = f"{source_type}::{chunk_id}"
            documents.append(
                _IndexedDocument(
                    vector_id=vector_id,
                    source_type=source_type,
                    input_file=str(source_file),
                    chunk_id=chunk_id,
                    chunk_type=str(raw_chunk.get("chunk_type") or "unknown"),
                    text=text,
                    token_count=token_count,
                    metadata=metadata_obj,
                )
            )

        return documents, flags

    def _build_signature(
        self,
        *,
        document_id: str,
        snapshots: list[SourceSnapshot],
    ) -> str:
        parts = [
            f"document_id={document_id}",
            f"embedding_url={self.runtime_config.ollama_embedding_url}",
            f"embedding_model={self.runtime_config.ollama_embedding_model}",
            f"chroma_host={self.runtime_config.chroma_host}",
            f"chroma_port={self.runtime_config.chroma_port}",
            f"collection_prefix={self.runtime_config.chroma_collection_prefix}",
            f"distance_metric={self.runtime_config.chroma_distance_metric}",
        ]
        for snapshot in snapshots:
            parts.append(f"source={snapshot.source_type}")
            parts.append(f"file={snapshot.file_path or '-'}")
            parts.append(f"missing={snapshot.missing}")
            parts.append(
                f"mtime={snapshot.last_modified.isoformat() if snapshot.last_modified else '-'}"
            )
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _safe_delete_collection(self, *, document_id: str, collection_name: str) -> None:
        client = self._get_client(document_id=document_id)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            # Ignore if collection does not exist.
            pass

    def _create_collection(self, *, document_id: str, collection_name: str):
        client = self._get_client(document_id=document_id)
        metadata = {"hnsw:space": self.runtime_config.chroma_distance_metric}
        return client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=self._embedding_function,
        )

    def _to_chroma_metadata(self, document: _IndexedDocument) -> dict[str, str | int | float | bool]:
        meta: dict[str, str | int | float | bool] = {
            "source_type": document.source_type,
            "chunk_id": document.chunk_id,
            "chunk_type": document.chunk_type,
            "input_file": document.input_file,
        }
        if document.token_count is not None:
            meta["token_count"] = document.token_count
        return meta

    def _build_state(self, *, document_id: str) -> _IndexState:
        input_dir = self._input_dir_for_document(document_id=document_id)
        self._check_backend_health(document_id=document_id)

        source_files, snapshots, quality_flags = self._collect_source_files(document_id=document_id)
        all_documents: list[_IndexedDocument] = []
        chunks_by_source: dict[str, int] = {source_type: 0 for source_type, _ in SOURCE_DEFINITIONS}
        seen_vector_ids: set[str] = set()

        for source_type, source_file in source_files.items():
            if source_file is None:
                continue

            source_documents, source_flags = self._load_documents_from_source(
                source_type=source_type,
                source_file=source_file,
            )
            quality_flags.extend(source_flags)

            for doc in source_documents:
                if doc.vector_id in seen_vector_ids:
                    quality_flags.append(
                        RetrievalQualityFlag(
                            code="VECTOR_ID_DUPLICATE",
                            message="Duplicate vector id in input chunks; later record was skipped.",
                            severity="warning",
                            context={"vector_id": doc.vector_id},
                        )
                    )
                    continue
                seen_vector_ids.add(doc.vector_id)
                all_documents.append(doc)
                chunks_by_source[doc.source_type] += 1

        snapshot_map = {snapshot.source_type: snapshot for snapshot in snapshots}
        for source_type, count in chunks_by_source.items():
            snapshot = snapshot_map[source_type]
            snapshot.chunk_count = count
            if not snapshot.missing and count == 0:
                quality_flags.append(
                    RetrievalQualityFlag(
                        code="SOURCE_HAS_NO_VALID_CHUNKS",
                        message="Source file exists but did not contribute valid chunks.",
                        severity="warning",
                        context={"source_type": source_type, "file_path": snapshot.file_path},
                    )
                )

        if not all_documents:
            raise ValueError(
                f"No valid chunks available for vector indexing in {input_dir}."
            )

        collection_name = self._collection_name(document_id)
        self._safe_delete_collection(document_id=document_id, collection_name=collection_name)
        collection = self._create_collection(document_id=document_id, collection_name=collection_name)

        batch_size = self.runtime_config.embedding_batch_size
        for offset in range(0, len(all_documents), batch_size):
            batch = all_documents[offset : offset + batch_size]
            collection.upsert(
                ids=[doc.vector_id for doc in batch],
                documents=[doc.text for doc in batch],
                metadatas=[self._to_chroma_metadata(doc) for doc in batch],
            )

        built_at = datetime.now(timezone.utc)
        signature = self._build_signature(document_id=document_id, snapshots=snapshots)
        index_stats = IndexStats(
            document_id=document_id,
            total_chunks=len(all_documents),
            chunks_by_source=chunks_by_source,
            built_at=built_at,
            index_signature=signature,
            collection_name=collection_name,
            source_snapshots=snapshots,
        )
        documents_by_vector_id = {doc.vector_id: doc for doc in all_documents}

        return _IndexState(
            document_id=document_id,
            index_signature=signature,
            collection_name=collection_name,
            built_at=built_at,
            documents_by_vector_id=documents_by_vector_id,
            index_stats=index_stats,
            quality_flags=quality_flags,
        )

    def _latest_signature_for_document(self, *, document_id: str) -> str:
        _, snapshots, _ = self._collect_source_files(document_id=document_id)
        return self._build_signature(document_id=document_id, snapshots=snapshots)

    def rebuild_index(self, *, document_id: str) -> tuple[IndexStats, list[RetrievalQualityFlag]]:
        state = self._build_state(document_id=document_id)
        with self._lock:
            self._state = state
        return state.index_stats, state.quality_flags

    def _ensure_index(self, *, document_id: str, rebuild_if_stale: bool) -> _IndexState:
        with self._lock:
            current_state = self._state

        if current_state is None or current_state.document_id != document_id:
            self.rebuild_index(document_id=document_id)
            with self._lock:
                return self._state  # type: ignore[return-value]

        if rebuild_if_stale and self.runtime_config.auto_rebuild_on_search:
            latest_signature = self._latest_signature_for_document(document_id=document_id)
            if latest_signature != current_state.index_signature:
                self.rebuild_index(document_id=document_id)
                with self._lock:
                    return self._state  # type: ignore[return-value]

        return current_state

    def _score_from_distance(self, distance: float | None) -> float | None:
        if distance is None:
            return None
        return 1.0 / (1.0 + distance)

    def search(
        self,
        *,
        query: str,
        document_id: str,
        top_k: int | None = None,
        source_types: list[SourceType] | None = None,
        rebuild_if_stale: bool | None = None,
    ) -> SearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        effective_top_k = top_k if top_k is not None else self.runtime_config.top_k_default
        if effective_top_k < 1 or effective_top_k > self.runtime_config.top_k_max:
            raise ValueError(f"top_k must be between 1 and {self.runtime_config.top_k_max}")

        effective_rebuild = (
            rebuild_if_stale
            if rebuild_if_stale is not None
            else self.runtime_config.auto_rebuild_on_search
        )
        state = self._ensure_index(document_id=document_id, rebuild_if_stale=effective_rebuild)
        self._check_backend_health(document_id=document_id)

        collection = self._create_collection(
            document_id=document_id,
            collection_name=state.collection_name,
        )
        where: dict[str, Any] | None = None
        if source_types:
            if len(source_types) == 1:
                where = {"source_type": source_types[0]}
            else:
                where = {"source_type": {"$in": source_types}}

        query_result = collection.query(
            query_texts=[normalized_query],
            n_results=effective_top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids_by_query = query_result.get("ids") or [[]]
        distances_by_query = query_result.get("distances") or [[]]
        ids = ids_by_query[0] if ids_by_query else []
        distances = distances_by_query[0] if distances_by_query else []

        hits: list[VectorSearchHit] = []
        response_flags = list(state.quality_flags)

        for idx, vector_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            doc = state.documents_by_vector_id.get(vector_id)
            if doc is None:
                response_flags.append(
                    RetrievalQualityFlag(
                        code="VECTOR_ID_NOT_IN_STATE",
                        message="Vector id from Chroma could not be resolved in local index state.",
                        severity="warning",
                        context={"vector_id": vector_id},
                    )
                )
                continue

            hits.append(
                VectorSearchHit(
                    rank=len(hits) + 1,
                    score=self._score_from_distance(distance),
                    distance=distance,
                    source_type=doc.source_type,
                    chunk_id=doc.chunk_id,
                    chunk_type=doc.chunk_type,
                    text=doc.text,
                    token_count=doc.token_count,
                    metadata=doc.metadata,
                    input_file=doc.input_file,
                )
            )

        return SearchResponse(
            document_id=document_id,
            query=normalized_query,
            top_k=effective_top_k,
            hits=hits,
            index_stats=state.index_stats,
            quality_flags=response_flags,
        )

    def readiness(self) -> dict[str, Any]:
        backend_reachable = True
        error_message: str | None = None
        try:
            self._check_backend_health(document_id=self.settings.document_id)
        except VectorBackendUnavailableError as exc:
            backend_reachable = False
            error_message = str(exc)

        with self._lock:
            state = self._state

        payload: dict[str, Any] = {
            "status": "ready" if backend_reachable else "degraded",
            "backend_reachable": backend_reachable,
            "chroma_client_mode": self.runtime_config.chroma_client_mode,
        }
        if error_message:
            payload["backend_error"] = error_message
        if self._persistent_fallback_reason:
            payload["persistent_fallback"] = self._persistent_fallback_reason

        if state is None:
            payload["index_built"] = False
            return payload

        payload["index_built"] = True
        payload["document_id"] = state.document_id
        payload["total_chunks"] = state.index_stats.total_chunks
        payload["built_at"] = state.index_stats.built_at.isoformat()
        payload["collection_name"] = state.index_stats.collection_name
        return payload
