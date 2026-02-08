from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from rank_bm25 import BM25Okapi

from retrieval.sparse_retrieval.config import (
    SparseRetrievalRuntimeConfig,
    SparseRetrievalSettings,
    get_sparse_retrieval_config,
    get_sparse_retrieval_settings,
)
from retrieval.sparse_retrieval.models import (
    IndexStats,
    RetrievalQualityFlag,
    SearchHit,
    SearchResponse,
    SourceSnapshot,
    SourceType,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9ÄÖÜäöüß]+")
PARAGRAPH_PATTERN = re.compile(r"§\s*([0-9]+[a-zA-Z]?)")

# Intentionally compact stopword set to reduce BM25 noise while keeping domain terms.
GERMAN_STOPWORDS = {
    "aber",
    "als",
    "am",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "da",
    "dadurch",
    "daher",
    "darum",
    "das",
    "daß",
    "dass",
    "dein",
    "deine",
    "dem",
    "den",
    "der",
    "des",
    "dessen",
    "deshalb",
    "die",
    "dies",
    "dieser",
    "dieses",
    "doch",
    "dort",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "er",
    "es",
    "euer",
    "eure",
    "für",
    "hatte",
    "hatten",
    "hattest",
    "hattet",
    "hier",
    "hinter",
    "ich",
    "ihr",
    "ihre",
    "im",
    "in",
    "ist",
    "ja",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jener",
    "jenes",
    "jetzt",
    "kann",
    "kannst",
    "koennen",
    "können",
    "machen",
    "mein",
    "meine",
    "mit",
    "muss",
    "musst",
    "muessen",
    "müssen",
    "nach",
    "nachdem",
    "nein",
    "nicht",
    "nun",
    "oder",
    "seid",
    "sein",
    "seine",
    "sich",
    "sie",
    "sind",
    "soll",
    "sollen",
    "sollst",
    "sonst",
    "soweit",
    "sowie",
    "und",
    "unser",
    "unsere",
    "unter",
    "vom",
    "von",
    "vor",
    "wann",
    "warum",
    "was",
    "weiter",
    "wenn",
    "wer",
    "werde",
    "werden",
    "wie",
    "wieder",
    "wieso",
    "wir",
    "wird",
    "wirst",
    "wo",
    "woher",
    "wohin",
    "zu",
    "zum",
    "zur",
    "zwischen",
}

SOURCE_DEFINITIONS: tuple[tuple[SourceType, str], ...] = (
    ("pdf", "pdf_chunks_glob"),
    ("faq_so", "faq_so_chunks_glob"),
    ("faq_sb", "faq_sb_chunks_glob"),
)


@dataclass(frozen=True)
class _IndexedDocument:
    source_type: SourceType
    input_file: str
    chunk_id: str
    chunk_type: str
    text: str
    token_count: int | None
    metadata: dict[str, Any]
    tokens: list[str]


@dataclass(frozen=True)
class _IndexState:
    document_id: str
    index_signature: str
    built_at: datetime
    bm25: BM25Okapi
    documents: list[_IndexedDocument]
    index_stats: IndexStats
    quality_flags: list[RetrievalQualityFlag]


class SparseBM25Retriever:
    def __init__(
        self,
        *,
        settings: SparseRetrievalSettings | None = None,
        runtime_config: SparseRetrievalRuntimeConfig | None = None,
    ) -> None:
        self.settings = settings or get_sparse_retrieval_settings()
        self.runtime_config = runtime_config or get_sparse_retrieval_config(self.settings)
        self._lock = Lock()
        self._state: _IndexState | None = None

    def _input_dir_for_document(self, *, document_id: str) -> Path:
        if self.settings.input_dir is not None:
            return self.runtime_config.input_dir
        return (self.settings.data_root_path / document_id / "chunks").resolve()

    def _tokenize(self, text: str) -> list[str]:
        candidate = PARAGRAPH_PATTERN.sub(r" paragraf \1 ", text)
        if self.runtime_config.lowercase:
            candidate = candidate.lower()

        tokens: list[str] = []
        for token in TOKEN_PATTERN.findall(candidate):
            if not self.runtime_config.include_numeric_tokens and token.isdigit():
                continue
            if token.isdigit():
                tokens.append(token)
                continue
            if len(token) < self.runtime_config.min_token_length:
                continue
            if self.runtime_config.remove_stopwords and token in GERMAN_STOPWORDS:
                continue
            tokens.append(token)
        return tokens

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
                        file_path=None,
                        chunk_count=0,
                        missing=True,
                        last_modified=None,
                    )
                )
                flags.append(
                    RetrievalQualityFlag(
                        code="SOURCE_FILE_MISSING",
                        message="Expected source file for sparse retrieval was not found.",
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
                        chunk_count=0,
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

            tokens = self._tokenize(text)
            if not tokens:
                flags.append(
                    RetrievalQualityFlag(
                        code="CHUNK_EMPTY_AFTER_TOKENIZATION",
                        message="Chunk lost all terms during tokenization and was skipped.",
                        severity="info",
                        context={"source_type": source_type, "chunk_id": chunk_id},
                    )
                )
                continue

            token_count_raw = raw_chunk.get("token_count")
            token_count: int | None
            if isinstance(token_count_raw, int):
                token_count = token_count_raw
            else:
                token_count = None

            metadata = raw_chunk.get("metadata")
            metadata_obj = metadata if isinstance(metadata, dict) else {}

            documents.append(
                _IndexedDocument(
                    source_type=source_type,
                    input_file=str(source_file),
                    chunk_id=chunk_id,
                    chunk_type=str(raw_chunk.get("chunk_type") or "unknown"),
                    text=text,
                    token_count=token_count,
                    metadata=metadata_obj,
                    tokens=tokens,
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
            f"k1={self.runtime_config.bm25_k1}",
            f"b={self.runtime_config.bm25_b}",
            f"epsilon={self.runtime_config.bm25_epsilon}",
            f"lowercase={self.runtime_config.lowercase}",
            f"remove_stopwords={self.runtime_config.remove_stopwords}",
            f"min_token_length={self.runtime_config.min_token_length}",
            f"include_numeric_tokens={self.runtime_config.include_numeric_tokens}",
        ]
        for snapshot in snapshots:
            parts.append(f"source={snapshot.source_type}")
            parts.append(f"file={snapshot.file_path or '-'}")
            parts.append(f"missing={snapshot.missing}")
            parts.append(
                f"mtime={snapshot.last_modified.isoformat() if snapshot.last_modified else '-'}"
            )
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _build_state(self, *, document_id: str) -> _IndexState:
        input_dir = self._input_dir_for_document(document_id=document_id)
        source_files, snapshots, quality_flags = self._collect_source_files(document_id=document_id)
        documents: list[_IndexedDocument] = []
        seen_chunk_keys: set[tuple[SourceType, str]] = set()
        chunks_by_source: dict[str, int] = {source_type: 0 for source_type, _ in SOURCE_DEFINITIONS}

        for source_type, source_file in source_files.items():
            if source_file is None:
                continue

            source_documents, source_flags = self._load_documents_from_source(
                source_type=source_type,
                source_file=source_file,
            )
            quality_flags.extend(source_flags)

            for doc in source_documents:
                key = (doc.source_type, doc.chunk_id)
                if key in seen_chunk_keys:
                    quality_flags.append(
                        RetrievalQualityFlag(
                            code="CHUNK_ID_DUPLICATE",
                            message="Duplicate chunk_id inside source; later record was skipped.",
                            severity="warning",
                            context={"source_type": doc.source_type, "chunk_id": doc.chunk_id},
                        )
                    )
                    continue
                seen_chunk_keys.add(key)
                documents.append(doc)
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

        if not documents:
            raise ValueError(
                f"No valid chunks available for BM25 indexing in {input_dir}."
            )

        corpus = [doc.tokens for doc in documents]
        bm25 = BM25Okapi(
            corpus=corpus,
            k1=self.runtime_config.bm25_k1,
            b=self.runtime_config.bm25_b,
            epsilon=self.runtime_config.bm25_epsilon,
        )

        built_at = datetime.now(timezone.utc)
        signature = self._build_signature(document_id=document_id, snapshots=snapshots)
        index_stats = IndexStats(
            document_id=document_id,
            total_chunks=len(documents),
            chunks_by_source=chunks_by_source,
            built_at=built_at,
            index_signature=signature,
            source_snapshots=snapshots,
        )

        return _IndexState(
            document_id=document_id,
            index_signature=signature,
            built_at=built_at,
            bm25=bm25,
            documents=documents,
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

    def _ensure_index(
        self,
        *,
        document_id: str,
        rebuild_if_stale: bool,
    ) -> _IndexState:
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
        query_tokens = self._tokenize(normalized_query)
        response_flags = list(state.quality_flags)

        if not query_tokens:
            response_flags.append(
                RetrievalQualityFlag(
                    code="QUERY_EMPTY_AFTER_TOKENIZATION",
                    message="Query contains no retrievable terms after tokenization.",
                    severity="warning",
                )
            )
            return SearchResponse(
                document_id=document_id,
                query=normalized_query,
                top_k=effective_top_k,
                hits=[],
                index_stats=state.index_stats,
                quality_flags=response_flags,
            )

        scores = state.bm25.get_scores(query_tokens)
        allowed_sources = set(source_types) if source_types else None

        ranked_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
        hits: list[SearchHit] = []
        for doc_index in ranked_indices:
            score = float(scores[doc_index])
            if score <= 0:
                continue

            doc = state.documents[doc_index]
            if allowed_sources is not None and doc.source_type not in allowed_sources:
                continue

            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=score,
                    source_type=doc.source_type,
                    chunk_id=doc.chunk_id,
                    chunk_type=doc.chunk_type,
                    text=doc.text,
                    token_count=doc.token_count,
                    metadata=doc.metadata,
                    input_file=doc.input_file,
                )
            )
            if len(hits) >= effective_top_k:
                break

        return SearchResponse(
            document_id=document_id,
            query=normalized_query,
            top_k=effective_top_k,
            hits=hits,
            index_stats=state.index_stats,
            quality_flags=response_flags,
        )

    def readiness(self) -> dict[str, Any]:
        with self._lock:
            state = self._state

        if state is None:
            return {"status": "ready", "index_built": False}

        return {
            "status": "ready",
            "index_built": True,
            "document_id": state.document_id,
            "total_chunks": state.index_stats.total_chunks,
            "built_at": state.index_stats.built_at.isoformat(),
        }
