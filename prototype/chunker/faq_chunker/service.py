from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken

from chunker.faq_chunker.config import (
    FAQChunkerRuntimeConfig,
    FAQChunkerSettings,
    get_faq_chunker_config,
    get_faq_chunker_settings,
)
from chunker.faq_chunker.models import (
    ChunkQualityFlag,
    ChunkingResult,
    ChunkingStats,
    FAQChunk,
    FAQChunkMetadata,
)

QUESTION_LIST_KEYS = ("questions", "faqs", "items", "entries", "data")
SO_FILENAME_MARKERS = ("so", "studienordnung", "faq-pdf", "pdf-faq")
SB_FILENAME_MARKERS = ("sb", "studienberatung", "beratung", "faq-studienberatung", "studienberatung-faq", "faq-sb", "sb-faq")


def _encode(encoder: tiktoken.Encoding, text: str) -> list[int]:
    return encoder.encode(text)


def _normalize_faq_source(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    if not normalized:
        raise ValueError("faq_source must not be empty")
    if re.fullmatch(r"[a-z0-9_]+", normalized) is None:
        raise ValueError("faq_source must match [a-z0-9_]+")
    return normalized


def _source_markers(faq_source: str) -> tuple[str, ...]:
    if faq_source == "so":
        return SO_FILENAME_MARKERS
    if faq_source == "sb":
        return SB_FILENAME_MARKERS
    return (faq_source,)


def _path_has_any_marker(path: Path, markers: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in markers)


def _sort_by_mtime_desc(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)


def _normalize_faq_id(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return str(raw_value)
    text = str(raw_value).strip()
    if not text:
        return None
    if text.isdigit():
        return str(int(text))
    return text


def _build_chunk_id(*, faq_id: str, faq_source: str) -> str:
    if faq_id.isdigit():
        return f"faq_{faq_source}_{int(faq_id):04d}"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", faq_id).strip("_").lower()
    if not safe:
        safe = "unknown"
    return f"faq_{faq_source}_{safe}"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_paragraph_label(section: str | None) -> str | None:
    if not section:
        return None
    match = re.search(r"§\s*\d+[a-zA-Z]?(?:\s*-\s*\d+[a-zA-Z]?)?", section)
    return match.group(0) if match else None


def _parse_page(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("page must be >= 1")
    return parsed


def _resolve_faq_payload(payload: dict | list) -> tuple[str, dict[str, Any], list[Any]]:
    if isinstance(payload, list):
        return "", {}, payload

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object or a list.")

    source_file = _normalize_text(payload.get("source_file"))
    raw_context = payload.get("context")
    context = raw_context if isinstance(raw_context, dict) else {}

    for key in QUESTION_LIST_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return source_file, context, candidate

    raise ValueError(f"Input JSON does not contain a FAQ list key. Expected one of: {QUESTION_LIST_KEYS}")


def _resolve_input_file(
    request_input_file: str | None,
    document_id: str,
    faq_source: str,
    settings: FAQChunkerSettings,
    runtime_config: FAQChunkerRuntimeConfig,
) -> Path:
    if request_input_file:
        candidate = Path(request_input_file)
        search_roots = [
            candidate,
            settings.project_root / candidate,
            settings.data_root_path / candidate,
            runtime_config.input_dir / candidate,
        ]
        for path in search_roots:
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(f"Input JSON not found: {request_input_file}")

    all_candidates = _sort_by_mtime_desc(list(runtime_config.input_dir.glob(runtime_config.input_glob)))
    if not all_candidates:
        raise FileNotFoundError(
            f"No FAQ input found in {runtime_config.input_dir} matching {runtime_config.input_glob}"
        )

    document_id_lc = document_id.lower()
    source_markers = _source_markers(faq_source)
    opposite_markers = SB_FILENAME_MARKERS if faq_source == "so" else SO_FILENAME_MARKERS

    document_candidates = [path for path in all_candidates if document_id_lc in path.name.lower()]
    search_pool = document_candidates if document_candidates else all_candidates

    strict_source_candidates = [
        path for path in search_pool if _path_has_any_marker(path, source_markers)
    ]
    strict_without_opposite = [
        path for path in strict_source_candidates if not _path_has_any_marker(path, opposite_markers)
    ]
    if strict_without_opposite:
        return strict_without_opposite[0].resolve()
    if strict_source_candidates:
        return strict_source_candidates[0].resolve()

    if faq_source == "so":
        # Backward-compatible fallback for existing study-order FAQ files
        # whose filenames may not include explicit source markers.
        generic_without_sb = [
            path for path in search_pool if not _path_has_any_marker(path, SB_FILENAME_MARKERS)
        ]
        if generic_without_sb:
            return generic_without_sb[0].resolve()
        return search_pool[0].resolve()

    raise FileNotFoundError(
        "No FAQ input found for faq_source "
        f"'{faq_source}' in {runtime_config.input_dir}. "
        "Provide input_file explicitly or include a source marker in filename "
        f"(expected one of: {source_markers})."
    )


def _resolve_output_dir(
    raw_value: str | None,
    settings: FAQChunkerSettings,
    runtime_config: FAQChunkerRuntimeConfig,
) -> Path:
    if not raw_value:
        return runtime_config.output_dir.resolve()

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (settings.project_root / candidate).resolve()


def _persist_result(
    result: ChunkingResult,
    output_dir: Path,
    runtime_config: FAQChunkerRuntimeConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{result.document_id}-{runtime_config.output_suffix}-{result.faq_source}"

    if not runtime_config.keep_previous_outputs:
        for existing in output_dir.glob(f"{filename_prefix}-*.json"):
            existing.unlink(missing_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{filename_prefix}-{timestamp}.json"
    payload = result.model_dump(mode="json")
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=runtime_config.json_indent),
        encoding="utf-8",
    )
    return output_file


def run_faq_chunking(
    *,
    document_id: str,
    request_input_file: str | None = None,
    request_output_dir: str | None = None,
    request_faq_source: str | None = None,
    persist_result: bool = True,
    settings: FAQChunkerSettings | None = None,
    runtime_config: FAQChunkerRuntimeConfig | None = None,
) -> tuple[ChunkingResult, Path, Path | None]:
    resolved_settings = settings or get_faq_chunker_settings()
    resolved_config = runtime_config or get_faq_chunker_config(resolved_settings)
    faq_source = _normalize_faq_source(request_faq_source or resolved_config.faq_source)

    input_file = _resolve_input_file(
        request_input_file=request_input_file,
        document_id=document_id,
        faq_source=faq_source,
        settings=resolved_settings,
        runtime_config=resolved_config,
    )
    output_dir = _resolve_output_dir(request_output_dir, resolved_settings, resolved_config)

    payload = json.loads(input_file.read_text(encoding="utf-8-sig"))
    source_file, context, question_entries = _resolve_faq_payload(payload)
    encoder = tiktoken.get_encoding(resolved_config.tokenizer_encoding)

    chunks: list[FAQChunk] = []
    quality_flags: list[ChunkQualityFlag] = []
    token_counts: list[int] = []

    questions_total = len(question_entries)
    questions_processed = 0
    questions_skipped = 0
    seen_faq_ids: set[str] = set()
    faq_id_to_chunk_id: dict[str, str] = {}
    chunk_ids: set[str] = set()

    for entry_index, raw_entry in enumerate(question_entries):
        if not isinstance(raw_entry, dict):
            questions_skipped += 1
            quality_flags.append(
                ChunkQualityFlag(
                    code="QUESTION_ITEM_NOT_OBJECT",
                    message="FAQ item is not an object and was skipped.",
                    severity="warning",
                    context={"entry_index": entry_index},
                )
            )
            continue

        faq_id = _normalize_faq_id(raw_entry.get("id"))
        if faq_id is None:
            questions_skipped += 1
            quality_flags.append(
                ChunkQualityFlag(
                    code="FAQ_ID_MISSING",
                    message="FAQ item has no valid id and was skipped.",
                    severity="warning",
                    context={"entry_index": entry_index},
                )
            )
            continue

        if faq_id in seen_faq_ids:
            questions_skipped += 1
            quality_flags.append(
                ChunkQualityFlag(
                    code="FAQ_ID_DUPLICATE",
                    message="Duplicate FAQ id found. Later entry was skipped.",
                    severity="warning",
                    context={"faq_id": faq_id, "entry_index": entry_index},
                )
            )
            continue

        question = _normalize_text(raw_entry.get("question"))
        answer = _normalize_text(raw_entry.get("answer"))
        if not question or not answer:
            questions_skipped += 1
            quality_flags.append(
                ChunkQualityFlag(
                    code="FAQ_EMPTY_QUESTION_OR_ANSWER",
                    message="FAQ entry has empty question or answer and was skipped.",
                    severity="warning",
                    context={"faq_id": faq_id, "entry_index": entry_index},
                )
            )
            continue

        try:
            page = _parse_page(raw_entry.get("page"))
        except ValueError:
            page = None
            quality_flags.append(
                ChunkQualityFlag(
                    code="FAQ_PAGE_INVALID",
                    message="FAQ page field is invalid and was set to null.",
                    severity="info",
                    context={"faq_id": faq_id, "raw_page": raw_entry.get("page")},
                )
            )

        previous_question_id = _normalize_faq_id(raw_entry.get("previous_question"))
        next_question_id = _normalize_faq_id(raw_entry.get("next_question"))
        section = _normalize_text(raw_entry.get("section")) or None
        paragraph_label = _extract_paragraph_label(section)

        chunk_id = _build_chunk_id(faq_id=faq_id, faq_source=faq_source)
        if chunk_id in chunk_ids:
            questions_skipped += 1
            quality_flags.append(
                ChunkQualityFlag(
                    code="CHUNK_ID_COLLISION",
                    message="Generated chunk_id is not unique. Entry was skipped.",
                    severity="error",
                    context={"faq_id": faq_id, "chunk_id": chunk_id},
                )
            )
            continue

        chunk_text = f"Frage: {question}\nAntwort: {answer}"
        token_count = len(_encode(encoder, chunk_text))

        metadata = FAQChunkMetadata(
            document_id=document_id,
            source_file=source_file,
            faq_source=faq_source,
            faq_id=faq_id,
            section=section,
            paragraph_label=paragraph_label,
            page=page,
            chunk_index=len(chunks),
            previous_question_id=previous_question_id,
            next_question_id=next_question_id,
        )

        chunks.append(
            FAQChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                token_count=token_count,
                metadata=metadata,
                flags=[],
            )
        )
        token_counts.append(token_count)
        questions_processed += 1
        seen_faq_ids.add(faq_id)
        faq_id_to_chunk_id[faq_id] = chunk_id
        chunk_ids.add(chunk_id)

    for index, chunk in enumerate(chunks):
        chunk.metadata.prev_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
        chunk.metadata.next_chunk_id = chunks[index + 1].chunk_id if index < len(chunks) - 1 else None

    for chunk in chunks:
        prev_question_id = chunk.metadata.previous_question_id
        next_question_id = chunk.metadata.next_question_id

        if prev_question_id is not None:
            prev_chunk_id = faq_id_to_chunk_id.get(prev_question_id)
            if prev_chunk_id is not None:
                chunk.metadata.previous_question_chunk_id = prev_chunk_id
            else:
                quality_flags.append(
                    ChunkQualityFlag(
                        code="PREVIOUS_QUESTION_REFERENCE_MISSING",
                        message="previous_question reference could not be resolved to a chunk id.",
                        severity="info",
                        context={
                            "chunk_id": chunk.chunk_id,
                            "faq_id": chunk.metadata.faq_id,
                            "previous_question_id": prev_question_id,
                        },
                    )
                )

        if next_question_id is not None:
            next_chunk_id = faq_id_to_chunk_id.get(next_question_id)
            if next_chunk_id is not None:
                chunk.metadata.next_question_chunk_id = next_chunk_id
            else:
                quality_flags.append(
                    ChunkQualityFlag(
                        code="NEXT_QUESTION_REFERENCE_MISSING",
                        message="next_question reference could not be resolved to a chunk id.",
                        severity="info",
                        context={
                            "chunk_id": chunk.chunk_id,
                            "faq_id": chunk.metadata.faq_id,
                            "next_question_id": next_question_id,
                        },
                    )
                )

    if questions_total > 0 and questions_processed == 0:
        quality_flags.append(
            ChunkQualityFlag(
                code="ALL_QUESTIONS_SKIPPED",
                message="No valid FAQ entries could be converted into chunks.",
                severity="error",
            )
        )

    stats = ChunkingStats.from_token_counts(
        token_counts,
        questions_total=questions_total,
        questions_processed=questions_processed,
        questions_skipped=questions_skipped,
    )

    result = ChunkingResult(
        document_id=document_id,
        faq_source=faq_source,
        input_file=str(input_file),
        source_file=source_file,
        context=context,
        chunks=chunks,
        stats=stats,
        quality_flags=quality_flags,
        created_at=datetime.now(timezone.utc),
    )

    output_file: Path | None = None
    if persist_result:
        output_file = _persist_result(result=result, output_dir=output_dir, runtime_config=resolved_config)

    return result, input_file, output_file
