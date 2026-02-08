from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal

from retrieval.hybrid_retrieval.config import (
    HybridRetrievalRuntimeConfig,
    HybridRetrievalSettings,
    get_hybrid_retrieval_config,
    get_hybrid_retrieval_settings,
)
from retrieval.hybrid_retrieval.models import (
    BackendIndexStats,
    BackendName,
    HybridIndexStats,
    HybridSearchHit,
    RetrievalQualityFlag,
    SearchResponse,
    SourceSnapshot,
    SourceType,
)

VALID_SOURCE_TYPES: set[str] = {"pdf", "faq_so", "faq_sb"}


class HybridBackendUnavailableError(RuntimeError):
    """Raised when no retrieval backend is currently available."""


@dataclass(frozen=True)
class _BackendCallResult:
    backend: BackendName
    ok: bool
    status_code: int | None
    payload: dict[str, Any] | None
    error: str | None


@dataclass
class _FusedCandidate:
    source_type: SourceType
    chunk_id: str
    chunk_type: str
    text: str
    token_count: int | None
    metadata: dict[str, Any]
    input_file: str
    rrf_score: float = 0.0

    sparse_rank: int | None = None
    sparse_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    vector_distance: float | None = None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_source_type(value: Any) -> SourceType | None:
    if value is None:
        return None
    token = str(value).strip()
    if token not in VALID_SOURCE_TYPES:
        return None
    return token  # type: ignore[return-value]


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    token = value.strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(token)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class HybridRetriever:
    def __init__(
        self,
        *,
        settings: HybridRetrievalSettings | None = None,
        runtime_config: HybridRetrievalRuntimeConfig | None = None,
    ) -> None:
        self.settings = settings or get_hybrid_retrieval_settings()
        self.runtime_config = runtime_config or get_hybrid_retrieval_config(self.settings)
        self._lock = Lock()
        self._last_index_stats: HybridIndexStats | None = None

    def _backend_base_url(self, backend: BackendName) -> str:
        if backend == "sparse":
            return self.runtime_config.sparse_base_url
        return self.runtime_config.vector_base_url

    def _request_json(
        self,
        *,
        method: Literal["GET", "POST"],
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request_obj = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(
                request_obj,
                timeout=self.runtime_config.http_timeout_seconds,
            ) as response:
                raw_body = response.read()
                if not raw_body:
                    return int(response.status), {}, None
                body = json.loads(raw_body.decode("utf-8"))
                if not isinstance(body, dict):
                    return int(response.status), None, "Backend response is not a JSON object."
                return int(response.status), body, None
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            detail = body_text or f"HTTP {exc.code}"
            try:
                decoded = json.loads(body_text)
                if isinstance(decoded, dict) and "detail" in decoded:
                    detail = str(decoded["detail"])
            except json.JSONDecodeError:
                pass
            return int(exc.code), None, detail
        except urllib.error.URLError as exc:
            return None, None, str(exc.reason)
        except socket.timeout:
            return None, None, "request timed out"
        except TimeoutError:
            return None, None, "request timed out"
        except Exception as exc:  # pragma: no cover - runtime safety net
            return None, None, str(exc)

    def _call_backend(
        self,
        *,
        backend: BackendName,
        endpoint: str,
        method: Literal["GET", "POST"],
        payload: dict[str, Any] | None = None,
    ) -> _BackendCallResult:
        base_url = self._backend_base_url(backend)
        url = f"{base_url}{endpoint}"
        status_code, response_payload, error_message = self._request_json(
            method=method,
            url=url,
            payload=payload,
        )
        ok = (
            status_code is not None
            and 200 <= status_code < 300
            and isinstance(response_payload, dict)
        )
        return _BackendCallResult(
            backend=backend,
            ok=ok,
            status_code=status_code,
            payload=response_payload if ok else None,
            error=error_message if not ok else None,
        )

    def _call_both_backends(
        self,
        *,
        endpoint: str,
        method: Literal["GET", "POST"],
        payload: dict[str, Any] | None = None,
    ) -> dict[BackendName, _BackendCallResult]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sparse_future = pool.submit(
                self._call_backend,
                backend="sparse",
                endpoint=endpoint,
                method=method,
                payload=payload,
            )
            vector_future = pool.submit(
                self._call_backend,
                backend="vector",
                endpoint=endpoint,
                method=method,
                payload=payload,
            )

            sparse_result = sparse_future.result()
            vector_result = vector_future.result()

        return {"sparse": sparse_result, "vector": vector_result}

    def _backend_failure_flag(self, result: _BackendCallResult) -> RetrievalQualityFlag:
        if result.status_code is None:
            message = "Retrieval backend is unreachable."
            code = "BACKEND_UNREACHABLE"
        else:
            message = "Retrieval backend returned a non-success status."
            code = "BACKEND_HTTP_ERROR"

        return RetrievalQualityFlag(
            code=code,
            message=message,
            severity="warning",
            context={
                "backend": result.backend,
                "status_code": result.status_code,
                "error": result.error,
            },
        )

    def _extract_backend_flags(self, result: _BackendCallResult) -> list[RetrievalQualityFlag]:
        if not result.ok or result.payload is None:
            return []

        raw_flags = result.payload.get("quality_flags")
        if not isinstance(raw_flags, list):
            return []

        flags: list[RetrievalQualityFlag] = []
        prefix = result.backend.upper()
        for entry in raw_flags:
            if not isinstance(entry, dict):
                continue
            code = str(entry.get("code") or "BACKEND_FLAG")
            message = str(entry.get("message") or "Backend provided a quality flag.")
            severity = str(entry.get("severity") or "warning")
            if severity not in {"info", "warning", "error"}:
                severity = "warning"
            raw_context = entry.get("context")
            context = raw_context if isinstance(raw_context, dict) else {}
            flags.append(
                RetrievalQualityFlag(
                    code=f"{prefix}_{code}",
                    message=message,
                    severity=severity,  # type: ignore[arg-type]
                    context={"backend": result.backend, **context},
                )
            )
        return flags

    def _backend_document_id_mismatch_flags(
        self,
        *,
        expected_document_id: str,
        result: _BackendCallResult,
    ) -> list[RetrievalQualityFlag]:
        if not result.ok or result.payload is None:
            return []

        observed_ids: set[str] = set()
        payload_document_id = result.payload.get("document_id")
        if payload_document_id is not None:
            observed_ids.add(str(payload_document_id).strip())

        raw_index_stats = result.payload.get("index_stats")
        if isinstance(raw_index_stats, dict):
            index_document_id = raw_index_stats.get("document_id")
            if index_document_id is not None:
                observed_ids.add(str(index_document_id).strip())

        flags: list[RetrievalQualityFlag] = []
        for observed in sorted(observed_ids):
            if observed and observed != expected_document_id:
                flags.append(
                    RetrievalQualityFlag(
                        code="BACKEND_DOCUMENT_ID_MISMATCH",
                        message="Backend responded for a different document_id.",
                        severity="warning",
                        context={
                            "backend": result.backend,
                            "expected_document_id": expected_document_id,
                            "observed_document_id": observed,
                        },
                    )
                )
        return flags

    def _parse_source_snapshots(self, value: Any) -> list[SourceSnapshot]:
        if not isinstance(value, list):
            return []

        snapshots: list[SourceSnapshot] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            source_type = _normalize_source_type(item.get("source_type"))
            if source_type is None:
                continue

            chunk_count = _coerce_int(item.get("chunk_count"))
            snapshots.append(
                SourceSnapshot(
                    source_type=source_type,
                    file_path=str(item.get("file_path")) if item.get("file_path") else None,
                    chunk_count=chunk_count if chunk_count is not None and chunk_count >= 0 else 0,
                    missing=bool(item.get("missing", False)),
                    last_modified=_parse_datetime(item.get("last_modified")),
                )
            )
        return snapshots

    def _build_backend_index_stats(self, result: _BackendCallResult) -> BackendIndexStats:
        if not result.ok or result.payload is None:
            return BackendIndexStats(
                backend=result.backend,
                available=False,
                status_code=result.status_code,
                error=result.error,
            )

        raw_index_stats = result.payload.get("index_stats")
        if not isinstance(raw_index_stats, dict):
            return BackendIndexStats(
                backend=result.backend,
                available=True,
                status_code=result.status_code,
                error=None,
            )

        raw_chunks_by_source = raw_index_stats.get("chunks_by_source")
        chunks_by_source = raw_chunks_by_source if isinstance(raw_chunks_by_source, dict) else {}
        total_chunks = _coerce_int(raw_index_stats.get("total_chunks"))

        normalized_chunks_by_source: dict[str, int] = {}
        for key, value in chunks_by_source.items():
            parsed = _coerce_int(value)
            if parsed is None or parsed < 0:
                continue
            normalized_chunks_by_source[str(key)] = parsed

        return BackendIndexStats(
            backend=result.backend,
            available=True,
            status_code=result.status_code,
            error=None,
            total_chunks=total_chunks,
            chunks_by_source=normalized_chunks_by_source,
            built_at=_parse_datetime(raw_index_stats.get("built_at")),
            index_signature=str(raw_index_stats.get("index_signature") or "") or None,
            collection_name=str(raw_index_stats.get("collection_name") or "") or None,
            source_snapshots=self._parse_source_snapshots(raw_index_stats.get("source_snapshots")),
        )

    def _build_hybrid_signature(
        self,
        *,
        document_id: str,
        sparse_stats: BackendIndexStats,
        vector_stats: BackendIndexStats,
    ) -> str:
        parts = [
            f"document_id={document_id}",
            f"rrf_rank_constant={self.runtime_config.rrf_rank_constant}",
            f"rank_window_size={self.runtime_config.rank_window_size}",
            f"sparse_weight={self.runtime_config.sparse_weight}",
            f"vector_weight={self.runtime_config.vector_weight}",
            f"sparse_available={sparse_stats.available}",
            f"sparse_signature={sparse_stats.index_signature or '-'}",
            f"vector_available={vector_stats.available}",
            f"vector_signature={vector_stats.index_signature or '-'}",
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _build_index_stats(
        self,
        *,
        document_id: str,
        sparse_result: _BackendCallResult,
        vector_result: _BackendCallResult,
    ) -> HybridIndexStats:
        sparse_stats = self._build_backend_index_stats(sparse_result)
        vector_stats = self._build_backend_index_stats(vector_result)
        fused_from: list[BackendName] = []
        if sparse_result.ok:
            fused_from.append("sparse")
        if vector_result.ok:
            fused_from.append("vector")

        return HybridIndexStats(
            document_id=document_id,
            built_at=datetime.now(timezone.utc),
            index_signature=self._build_hybrid_signature(
                document_id=document_id,
                sparse_stats=sparse_stats,
                vector_stats=vector_stats,
            ),
            fused_from_backends=fused_from,
            sparse=sparse_stats,
            vector=vector_stats,
        )

    def _empty_candidate_from_hit(
        self,
        *,
        source_type: SourceType,
        chunk_id: str,
        raw_hit: dict[str, Any],
    ) -> _FusedCandidate:
        raw_metadata = raw_hit.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        token_count = _coerce_int(raw_hit.get("token_count"))
        return _FusedCandidate(
            source_type=source_type,
            chunk_id=chunk_id,
            chunk_type=str(raw_hit.get("chunk_type") or "unknown"),
            text=str(raw_hit.get("text") or ""),
            token_count=token_count,
            metadata=metadata,
            input_file=str(raw_hit.get("input_file") or ""),
        )

    def _merge_candidate_data(self, candidate: _FusedCandidate, raw_hit: dict[str, Any]) -> None:
        if not candidate.text:
            candidate.text = str(raw_hit.get("text") or "")
        if not candidate.chunk_type or candidate.chunk_type == "unknown":
            candidate.chunk_type = str(raw_hit.get("chunk_type") or "unknown")
        if candidate.token_count is None:
            candidate.token_count = _coerce_int(raw_hit.get("token_count"))
        if not candidate.input_file:
            candidate.input_file = str(raw_hit.get("input_file") or "")
        if not candidate.metadata:
            raw_metadata = raw_hit.get("metadata")
            if isinstance(raw_metadata, dict):
                candidate.metadata = raw_metadata

    def _ingest_hits(
        self,
        *,
        backend: BackendName,
        hits: Any,
        weight: float,
        fused: dict[tuple[SourceType, str], _FusedCandidate],
        quality_flags: list[RetrievalQualityFlag],
    ) -> int:
        if not isinstance(hits, list):
            quality_flags.append(
                RetrievalQualityFlag(
                    code="BACKEND_HITS_INVALID",
                    message="Backend hits payload is not a list.",
                    severity="warning",
                    context={"backend": backend},
                )
            )
            return 0

        valid_hits = 0
        for fallback_rank, raw_hit in enumerate(hits, start=1):
            if not isinstance(raw_hit, dict):
                quality_flags.append(
                    RetrievalQualityFlag(
                        code="BACKEND_HIT_INVALID",
                        message="Backend hit entry is not an object and was ignored.",
                        severity="info",
                        context={"backend": backend, "position": fallback_rank},
                    )
                )
                continue

            source_type = _normalize_source_type(raw_hit.get("source_type"))
            chunk_id = str(raw_hit.get("chunk_id") or "").strip()
            if source_type is None or not chunk_id:
                quality_flags.append(
                    RetrievalQualityFlag(
                        code="BACKEND_HIT_KEY_INVALID",
                        message="Backend hit misses source_type/chunk_id and was ignored.",
                        severity="info",
                        context={"backend": backend, "position": fallback_rank},
                    )
                )
                continue

            rank = _coerce_int(raw_hit.get("rank"))
            if rank is None or rank < 1:
                rank = fallback_rank

            fusion_key = (source_type, chunk_id)
            candidate = fused.get(fusion_key)
            if candidate is None:
                candidate = self._empty_candidate_from_hit(
                    source_type=source_type,
                    chunk_id=chunk_id,
                    raw_hit=raw_hit,
                )
                fused[fusion_key] = candidate
            else:
                self._merge_candidate_data(candidate, raw_hit)

            candidate.rrf_score += weight * (1.0 / (self.runtime_config.rrf_rank_constant + rank))
            if backend == "sparse":
                candidate.sparse_rank = rank
                candidate.sparse_score = _coerce_float(raw_hit.get("score"))
            else:
                candidate.vector_rank = rank
                candidate.vector_score = _coerce_float(raw_hit.get("score"))
                candidate.vector_distance = _coerce_float(raw_hit.get("distance"))

            valid_hits += 1

        return valid_hits

    def _sorted_candidates(
        self,
        candidates: list[_FusedCandidate],
    ) -> list[_FusedCandidate]:
        def tie_rank(value: _FusedCandidate) -> int:
            ranks = [rank for rank in (value.sparse_rank, value.vector_rank) if rank is not None]
            return min(ranks) if ranks else 10**9

        return sorted(
            candidates,
            key=lambda item: (-item.rrf_score, tie_rank(item), item.source_type, item.chunk_id),
        )

    def rebuild_index(self, *, document_id: str) -> tuple[HybridIndexStats, list[RetrievalQualityFlag]]:
        payload = {"document_id": document_id}
        calls = self._call_both_backends(endpoint="/index/rebuild", method="POST", payload=payload)
        sparse_result = calls["sparse"]
        vector_result = calls["vector"]

        quality_flags: list[RetrievalQualityFlag] = []
        for result in (sparse_result, vector_result):
            if not result.ok:
                quality_flags.append(self._backend_failure_flag(result))
            quality_flags.extend(self._extract_backend_flags(result))
            quality_flags.extend(
                self._backend_document_id_mismatch_flags(
                    expected_document_id=document_id,
                    result=result,
                )
            )

        if not sparse_result.ok and not vector_result.ok:
            raise HybridBackendUnavailableError(
                "Neither sparse nor vector backend is currently available."
            )

        index_stats = self._build_index_stats(
            document_id=document_id,
            sparse_result=sparse_result,
            vector_result=vector_result,
        )

        with self._lock:
            self._last_index_stats = index_stats

        return index_stats, quality_flags

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
        backend_top_k = max(effective_top_k, self.runtime_config.rank_window_size)

        payload: dict[str, Any] = {
            "query": normalized_query,
            "document_id": document_id,
            "top_k": backend_top_k,
            "rebuild_if_stale": effective_rebuild,
        }
        if source_types:
            payload["source_types"] = source_types

        calls = self._call_both_backends(endpoint="/search", method="POST", payload=payload)
        sparse_result = calls["sparse"]
        vector_result = calls["vector"]

        quality_flags: list[RetrievalQualityFlag] = []
        for result in (sparse_result, vector_result):
            if not result.ok:
                quality_flags.append(self._backend_failure_flag(result))
            quality_flags.extend(self._extract_backend_flags(result))
            quality_flags.extend(
                self._backend_document_id_mismatch_flags(
                    expected_document_id=document_id,
                    result=result,
                )
            )

        if not sparse_result.ok and not vector_result.ok:
            raise HybridBackendUnavailableError(
                "Neither sparse nor vector backend is currently available."
            )

        fused: dict[tuple[SourceType, str], _FusedCandidate] = {}
        sparse_hit_count = 0
        vector_hit_count = 0

        if sparse_result.ok and sparse_result.payload is not None:
            sparse_hit_count = self._ingest_hits(
                backend="sparse",
                hits=sparse_result.payload.get("hits"),
                weight=self.runtime_config.sparse_weight,
                fused=fused,
                quality_flags=quality_flags,
            )
            if sparse_hit_count == 0:
                quality_flags.append(
                    RetrievalQualityFlag(
                        code="SPARSE_NO_HITS",
                        message="Sparse backend returned no valid hits.",
                        severity="info",
                    )
                )

        if vector_result.ok and vector_result.payload is not None:
            vector_hit_count = self._ingest_hits(
                backend="vector",
                hits=vector_result.payload.get("hits"),
                weight=self.runtime_config.vector_weight,
                fused=fused,
                quality_flags=quality_flags,
            )
            if vector_hit_count == 0:
                quality_flags.append(
                    RetrievalQualityFlag(
                        code="VECTOR_NO_HITS",
                        message="Vector backend returned no valid hits.",
                        severity="info",
                    )
                )

        ordered = self._sorted_candidates(list(fused.values()))
        selected = ordered[:effective_top_k]

        if not selected:
            quality_flags.append(
                RetrievalQualityFlag(
                    code="HYBRID_NO_HITS",
                    message="Hybrid fusion produced no hits.",
                    severity="warning",
                )
            )

        single_backend_hits = sum(
            1
            for hit in selected
            if (hit.sparse_rank is None) != (hit.vector_rank is None)
        )
        if single_backend_hits > 0:
            quality_flags.append(
                RetrievalQualityFlag(
                    code="HYBRID_SINGLE_BACKEND_EVIDENCE",
                    message="Some final hits are supported by only one retrieval backend.",
                    severity="info",
                    context={"count": single_backend_hits},
                )
            )

        hits = [
            HybridSearchHit(
                rank=index + 1,
                rrf_score=item.rrf_score,
                source_type=item.source_type,
                chunk_id=item.chunk_id,
                chunk_type=item.chunk_type,
                text=item.text,
                token_count=item.token_count,
                metadata=item.metadata,
                input_file=item.input_file,
                sparse_rank=item.sparse_rank,
                sparse_score=item.sparse_score,
                vector_rank=item.vector_rank,
                vector_score=item.vector_score,
                vector_distance=item.vector_distance,
            )
            for index, item in enumerate(selected)
        ]

        index_stats = self._build_index_stats(
            document_id=document_id,
            sparse_result=sparse_result,
            vector_result=vector_result,
        )
        with self._lock:
            self._last_index_stats = index_stats

        return SearchResponse(
            document_id=document_id,
            query=normalized_query,
            top_k=effective_top_k,
            hits=hits,
            index_stats=index_stats,
            quality_flags=quality_flags,
        )

    def readiness(self) -> dict[str, Any]:
        calls = self._call_both_backends(endpoint="/ready", method="GET", payload=None)
        sparse_result = calls["sparse"]
        vector_result = calls["vector"]

        backend_payloads: dict[str, Any] = {
            "sparse": {
                "available": sparse_result.ok,
                "status_code": sparse_result.status_code,
                "error": sparse_result.error,
            },
            "vector": {
                "available": vector_result.ok,
                "status_code": vector_result.status_code,
                "error": vector_result.error,
            },
        }

        if sparse_result.ok and sparse_result.payload is not None:
            backend_payloads["sparse"]["ready_payload"] = sparse_result.payload
        if vector_result.ok and vector_result.payload is not None:
            backend_payloads["vector"]["ready_payload"] = vector_result.payload

        overall_status = "ready" if (sparse_result.ok or vector_result.ok) else "degraded"
        response: dict[str, Any] = {"status": overall_status, "backends": backend_payloads}

        with self._lock:
            last_stats = self._last_index_stats
        if last_stats is not None:
            response["last_index_document_id"] = last_stats.document_id
            response["last_index_built_at"] = last_stats.built_at.isoformat()
            response["last_index_signature"] = last_stats.index_signature

        return response
