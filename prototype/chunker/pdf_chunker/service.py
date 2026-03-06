from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re

import tiktoken

from chunker.pdf_chunker.config import (
    PDFChunkerRuntimeConfig,
    PDFChunkerSettings,
    get_pdf_chunker_config,
    get_pdf_chunker_settings,
)
from chunker.pdf_chunker.models import (
    ChunkMetadata,
    ChunkQualityFlag,
    ChunkingResult,
    ChunkingStats,
    PDFChunk,
)

try:
    import syntok.segmenter as syntok_segmenter
except ImportError:  # pragma: no cover - environment dependent
    syntok_segmenter = None


def _join_sentence_tokens(sentence: list) -> str:
    return "".join(token.value + token.spacing for token in sentence).strip()


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []

    sentences: list[str] = []
    if syntok_segmenter is not None:
        for paragraph in syntok_segmenter.process(text):
            for sentence in paragraph:
                candidate = _join_sentence_tokens(sentence)
                if candidate:
                    sentences.append(candidate)
    else:
        # Regex fallback if syntok is not available in the runtime environment.
        rough = re.split(r"(?<=[.!?])\s+", text)
        sentences.extend([part.strip() for part in rough if part.strip()])

    if sentences:
        return sentences

    fallback = [line.strip() for line in text.splitlines() if line.strip()]
    return fallback if fallback else [text.strip()]


def _encode(encoder: tiktoken.Encoding, text: str) -> list[int]:
    return encoder.encode(text)


def _decode(encoder: tiktoken.Encoding, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return encoder.decode(token_ids)


def _split_oversized_sentence(
    sentence: str,
    encoder: tiktoken.Encoding,
    max_chunk_tokens: int,
) -> list[str]:
    token_ids = _encode(encoder, sentence)
    if len(token_ids) <= max_chunk_tokens:
        return [sentence]

    parts: list[str] = []
    cursor = 0
    while cursor < len(token_ids):
        slice_tokens = token_ids[cursor : cursor + max_chunk_tokens]
        parts.append(_decode(encoder, slice_tokens).strip())
        cursor += max_chunk_tokens
    return [part for part in parts if part]


def _build_main_text_chunks(
    text: str,
    encoder: tiktoken.Encoding,
    min_chunk_tokens: int,
    max_chunk_tokens: int,
    quality_flags: list[ChunkQualityFlag],
    section_id: str,
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        sentence_parts = _split_oversized_sentence(sentence, encoder, max_chunk_tokens)
        if len(sentence_parts) > 1:
            quality_flags.append(
                ChunkQualityFlag(
                    code="OVERSIZED_SENTENCE_SPLIT",
                    message="A sentence exceeded max_chunk_tokens and was split.",
                    severity="info",
                    context={"section_id": section_id, "parts": len(sentence_parts)},
                )
            )

        for piece in sentence_parts:
            if not current_sentences:
                current_sentences = [piece]
                continue

            candidate = " ".join([*current_sentences, piece])
            if len(_encode(encoder, candidate)) <= max_chunk_tokens:
                current_sentences.append(piece)
            else:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = [piece]

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    if len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        current_tokens = len(_encode(encoder, current))

        if current_tokens >= min_chunk_tokens or idx == len(chunks) - 1:
            if current_tokens < min_chunk_tokens and merged:
                candidate = f"{merged[-1]} {current}".strip()
                if len(_encode(encoder, candidate)) <= max_chunk_tokens:
                    merged[-1] = candidate
                else:
                    merged.append(current)
            else:
                merged.append(current)
            idx += 1
            continue

        candidate = f"{current} {chunks[idx + 1]}".strip()
        if len(_encode(encoder, candidate)) <= max_chunk_tokens:
            chunks[idx + 1] = candidate
            idx += 1
            continue

        merged.append(current)
        idx += 1

    return [chunk for chunk in merged if chunk]


def _apply_overlap_and_heading(
    *,
    section_chunks: list[str],
    encoder: tiktoken.Encoding,
    max_chunk_tokens: int,
    overlap_tokens: int,
    heading_prefix: str | None,
    quality_flags: list[ChunkQualityFlag],
    section_id: str,
) -> list[str]:
    if not section_chunks:
        return []

    heading_tokens = _encode(encoder, heading_prefix) if heading_prefix else []
    body_budget = max_chunk_tokens - len(heading_tokens)
    if body_budget <= 0:
        quality_flags.append(
            ChunkQualityFlag(
                code="HEADING_TOO_LONG",
                message="Heading prefix consumed complete token budget.",
                severity="error",
                context={"section_id": section_id},
            )
        )
        body_budget = max_chunk_tokens

    final_chunks: list[str] = []
    prev_tokens: list[int] = []

    for index, base_chunk in enumerate(section_chunks):
        current_tokens = _encode(encoder, base_chunk)
        if len(current_tokens) > body_budget:
            current_tokens = current_tokens[:body_budget]
            quality_flags.append(
                ChunkQualityFlag(
                    code="CHUNK_TRIMMED_FOR_HEADING",
                    message="Chunk body was trimmed to fit heading prefix.",
                    severity="info",
                    context={"section_id": section_id, "index": index},
                )
            )

        prefix_tokens: list[int] = []
        if index > 0 and overlap_tokens > 0 and prev_tokens:
            capacity = body_budget - len(current_tokens)
            overlap_used = min(overlap_tokens, max(capacity, 0))
            if overlap_used > 0:
                prefix_tokens = prev_tokens[-overlap_used:]

        combined_tokens = [*prefix_tokens, *current_tokens]

        if len(combined_tokens) > body_budget:
            combined_tokens = combined_tokens[:body_budget]
        final_tokens = [*heading_tokens, *combined_tokens] if heading_tokens else combined_tokens[:max_chunk_tokens]

        final_text = _decode(encoder, final_tokens).strip()
        if final_text:
            final_chunks.append(final_text)
        prev_tokens = current_tokens

    return final_chunks


def _serialize_table_rows(rows: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        clean_cells = [cell.strip() for cell in row if cell and cell.strip()]
        if not clean_cells:
            continue
        lines.append(" | ".join(clean_cells))
    return lines


def _build_table_chunks(
    *,
    table: dict,
    section: dict | None,
    encoder: tiktoken.Encoding,
    max_chunk_tokens: int,
    heading_prefix: str | None,
    quality_flags: list[ChunkQualityFlag],
) -> list[str]:
    rows = table.get("rows") or []
    row_lines = _serialize_table_rows(rows)
    if not row_lines:
        return []

    heading_tokens = _encode(encoder, heading_prefix) if heading_prefix else []
    body_budget = max_chunk_tokens - len(heading_tokens)
    if body_budget <= 0:
        body_budget = max_chunk_tokens
        quality_flags.append(
            ChunkQualityFlag(
                code="TABLE_HEADING_TOO_LONG",
                message="Table heading prefix exceeded token budget.",
                severity="warning",
                context={"table_id": table.get("table_id")},
            )
        )

    chunks: list[str] = []
    current_lines: list[str] = []

    for row_line in row_lines:
        row_tokens = _encode(encoder, row_line)
        if len(row_tokens) > body_budget:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
            cursor = 0
            while cursor < len(row_tokens):
                piece_tokens = row_tokens[cursor : cursor + body_budget]
                piece = _decode(encoder, piece_tokens).strip()
                if piece:
                    chunks.append(piece)
                cursor += body_budget
            quality_flags.append(
                ChunkQualityFlag(
                    code="TABLE_ROW_SPLIT",
                    message="A table row exceeded token budget and was split.",
                    severity="info",
                    context={"table_id": table.get("table_id")},
                )
            )
            continue

        candidate = "\n".join([*current_lines, row_line]).strip()
        if len(_encode(encoder, candidate)) <= body_budget:
            current_lines.append(row_line)
        else:
            chunks.append("\n".join(current_lines))
            current_lines = [row_line]

    if current_lines:
        chunks.append("\n".join(current_lines))

    if not heading_tokens:
        return [chunk for chunk in chunks if chunk.strip()]

    prefixed: list[str] = []
    for chunk in chunks:
        body_tokens = _encode(encoder, chunk)
        body_tokens = body_tokens[:body_budget]
        merged = _decode(encoder, [*heading_tokens, *body_tokens]).strip()
        if merged:
            prefixed.append(merged)
    return prefixed


def _build_heading_path(
    section: dict,
    section_by_id: dict[str, dict],
) -> list[str]:
    path_nodes: list[dict] = []
    cursor: dict | None = section
    guard = 0
    while cursor is not None and guard < 20:
        path_nodes.append(cursor)
        parent_id = cursor.get("parent_section_id")
        cursor = section_by_id.get(parent_id) if parent_id else None
        guard += 1
    path_nodes.reverse()

    labels: list[str] = []
    for node in path_nodes:
        label = (node.get("label") or "").strip()
        title = (node.get("title") or "").strip()
        if label and title:
            labels.append(f"{label} {title}")
        elif label:
            labels.append(label)
        elif title:
            labels.append(title)
    return labels


def _build_chunk_metadata(
    *,
    document_id: str,
    source_file: str,
    section: dict | None,
    path_labels: list[str],
    chunk_index_in_section: int,
    table_id: str | None,
) -> ChunkMetadata:
    if section is None:
        return ChunkMetadata(
            document_id=document_id,
            source_file=source_file,
            section_id=None,
            section_kind=None,
            section_label=None,
            section_title=None,
            parent_section_id=None,
            path_labels=path_labels,
            start_page=None,
            end_page=None,
            table_ids=[],
            table_id=table_id,
            chunk_index_in_section=chunk_index_in_section,
        )

    return ChunkMetadata(
        document_id=document_id,
        source_file=source_file,
        section_id=section.get("section_id"),
        section_kind=section.get("kind"),
        section_label=section.get("label"),
        section_title=section.get("title"),
        parent_section_id=section.get("parent_section_id"),
        path_labels=path_labels,
        start_page=section.get("start_page"),
        end_page=section.get("end_page"),
        table_ids=section.get("table_ids") or [],
        table_id=table_id,
        chunk_index_in_section=chunk_index_in_section,
    )


def _resolve_input_file(
    request_input_file: str | None,
    document_id: str,
    settings: PDFChunkerSettings,
    runtime_config: PDFChunkerRuntimeConfig,
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

    preferred = sorted(
        runtime_config.input_dir.glob(f"{document_id}*pdf-extractor*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0].resolve()

    candidates = sorted(
        runtime_config.input_dir.glob(runtime_config.input_glob),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        f"No extractor output found in {runtime_config.input_dir} matching {runtime_config.input_glob}"
    )


def _resolve_output_dir(
    raw_value: str | None,
    settings: PDFChunkerSettings,
    runtime_config: PDFChunkerRuntimeConfig,
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
    runtime_config: PDFChunkerRuntimeConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{result.document_id}-{runtime_config.output_suffix}"

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


def run_pdf_chunking(
    *,
    document_id: str,
    request_input_file: str | None = None,
    request_output_dir: str | None = None,
    persist_result: bool = True,
    settings: PDFChunkerSettings | None = None,
    runtime_config: PDFChunkerRuntimeConfig | None = None,
) -> tuple[ChunkingResult, Path, Path | None]:
    resolved_settings = settings or get_pdf_chunker_settings()
    resolved_config = runtime_config or get_pdf_chunker_config(resolved_settings)
    input_file = _resolve_input_file(request_input_file, document_id, resolved_settings, resolved_config)
    output_dir = _resolve_output_dir(request_output_dir, resolved_settings, resolved_config)

    payload = json.loads(input_file.read_text(encoding="utf-8"))
    source_file = payload.get("source_file", "")
    context = payload.get("context") or {}
    sections = payload.get("sections") or []
    tables = payload.get("tables") or []
    section_by_id = {section["section_id"]: section for section in sections if section.get("section_id")}
    encoder = tiktoken.get_encoding(resolved_config.tokenizer_encoding)

    chunks: list[PDFChunk] = []
    quality_flags: list[ChunkQualityFlag] = []
    chunk_counter = 1
    sections_processed = 0
    sections_skipped = 0
    tables_processed = 0
    tables_skipped = 0
    next_chunk_index_by_section: dict[str | None, int] = {}

    def consume_chunk_index(section_key: str | None) -> int:
        value = next_chunk_index_by_section.get(section_key, 0)
        next_chunk_index_by_section[section_key] = value + 1
        return value

    if syntok_segmenter is None:
        quality_flags.append(
            ChunkQualityFlag(
                code="SYNTOK_UNAVAILABLE",
                message="syntok is not installed; regex sentence splitting fallback is active.",
                severity="warning",
            )
        )

    text_sections = [section for section in sections if section.get("kind") in {"paragraph", "annex"}]
    if not text_sections:
        quality_flags.append(
            ChunkQualityFlag(
                code="NO_TEXT_SECTIONS",
                message="No paragraph/annex sections were found in extractor input.",
                severity="warning",
            )
        )

    for section in text_sections:
        normalized_text = (section.get("normalized_text") or "").strip()
        if not normalized_text:
            sections_skipped += 1
            quality_flags.append(
                ChunkQualityFlag(
                    code="SECTION_EMPTY_SKIPPED",
                    message="Section has empty normalized_text and was skipped.",
                    severity="info",
                    context={"section_id": section.get("section_id")},
                )
            )
            continue

        sections_processed += 1
        path_labels = _build_heading_path(section, section_by_id)
        heading_prefix: str | None = None
        heading_token_count = 0
        if resolved_config.prepend_heading_path and path_labels:
            heading_prefix = " > ".join(path_labels).strip() + "\n\n"
            heading_token_count = len(_encode(encoder, heading_prefix))

        body_max_tokens = max(32, resolved_config.max_chunk_tokens - heading_token_count)
        body_min_tokens = max(1, min(resolved_config.min_chunk_tokens, body_max_tokens))

        section_quality_flags: list[ChunkQualityFlag] = []
        base_chunks = _build_main_text_chunks(
            text=normalized_text,
            encoder=encoder,
            min_chunk_tokens=body_min_tokens,
            max_chunk_tokens=body_max_tokens,
            quality_flags=section_quality_flags,
            section_id=section.get("section_id") or "unknown",
        )
        final_text_chunks = _apply_overlap_and_heading(
            section_chunks=base_chunks,
            encoder=encoder,
            max_chunk_tokens=resolved_config.max_chunk_tokens,
            overlap_tokens=resolved_config.overlap_tokens,
            heading_prefix=heading_prefix,
            quality_flags=section_quality_flags,
            section_id=section.get("section_id") or "unknown",
        )
        quality_flags.extend(section_quality_flags)

        prev_chunk_id: str | None = None
        section_key = section.get("section_id")
        for chunk_text in final_text_chunks:
            chunk_id = f"ch_{chunk_counter:06d}"
            chunk_counter += 1
            section_chunk_index = consume_chunk_index(section_key)
            metadata = _build_chunk_metadata(
                document_id=document_id,
                source_file=source_file,
                section=section,
                path_labels=path_labels,
                chunk_index_in_section=section_chunk_index,
                table_id=None,
            )
            metadata.prev_chunk_id = prev_chunk_id
            record = PDFChunk(
                chunk_id=chunk_id,
                chunk_type="text",
                text=chunk_text,
                token_count=len(_encode(encoder, chunk_text)),
                metadata=metadata,
                flags=[],
            )
            chunks.append(record)
            if prev_chunk_id is not None:
                chunks[-2].metadata.next_chunk_id = chunk_id
            prev_chunk_id = chunk_id

    if resolved_config.include_table_chunks:
        for table in tables:
            table_id = table.get("table_id")
            section_id = table.get("section_id")
            section = section_by_id.get(section_id) if section_id else None
            if section is None:
                quality_flags.append(
                    ChunkQualityFlag(
                        code="TABLE_WITHOUT_SECTION",
                        message="Table has no section assignment.",
                        severity="info",
                        context={"table_id": table_id, "page": table.get("page")},
                    )
                )
            path_labels = _build_heading_path(section, section_by_id) if section else []
            heading_prefix: str | None = None
            if resolved_config.prepend_heading_path and path_labels:
                heading_prefix = " > ".join(path_labels).strip() + "\n\n"

            table_quality_flags: list[ChunkQualityFlag] = []
            table_chunks = _build_table_chunks(
                table=table,
                section=section,
                encoder=encoder,
                max_chunk_tokens=resolved_config.max_chunk_tokens,
                heading_prefix=heading_prefix,
                quality_flags=table_quality_flags,
            )
            quality_flags.extend(table_quality_flags)
            if not table_chunks:
                tables_skipped += 1
                continue

            tables_processed += 1
            prev_chunk_id: str | None = None
            section_key = section.get("section_id") if section else None
            for chunk_text in table_chunks:
                chunk_id = f"ch_{chunk_counter:06d}"
                chunk_counter += 1
                section_chunk_index = consume_chunk_index(section_key)
                metadata = _build_chunk_metadata(
                    document_id=document_id,
                    source_file=source_file,
                    section=section,
                    path_labels=path_labels,
                    chunk_index_in_section=section_chunk_index,
                    table_id=table_id,
                )
                metadata.prev_chunk_id = prev_chunk_id
                record = PDFChunk(
                    chunk_id=chunk_id,
                    chunk_type="table",
                    text=chunk_text,
                    token_count=len(_encode(encoder, chunk_text)),
                    metadata=metadata,
                    flags=[],
                )
                chunks.append(record)
                if prev_chunk_id is not None:
                    chunks[-2].metadata.next_chunk_id = chunk_id
                prev_chunk_id = chunk_id

    token_counts = [chunk.token_count for chunk in chunks]
    text_below_min = sum(
        1
        for chunk in chunks
        if chunk.chunk_type == "text" and chunk.token_count < resolved_config.min_chunk_tokens
    )
    table_below_min = sum(
        1
        for chunk in chunks
        if chunk.chunk_type == "table" and chunk.token_count < resolved_config.min_chunk_tokens
    )
    if text_below_min:
        quality_flags.append(
            ChunkQualityFlag(
                code="TEXT_CHUNKS_BELOW_MIN",
                message="Some text chunks are below min_chunk_tokens (usually short sections).",
                severity="info",
                context={"count": text_below_min, "min_chunk_tokens": resolved_config.min_chunk_tokens},
            )
        )
    if table_below_min:
        quality_flags.append(
            ChunkQualityFlag(
                code="TABLE_CHUNKS_BELOW_MIN",
                message="Some table chunks are below min_chunk_tokens.",
                severity="info",
                context={"count": table_below_min, "min_chunk_tokens": resolved_config.min_chunk_tokens},
            )
        )

    stats = ChunkingStats.from_token_counts(
        token_counts,
        text_chunks=sum(1 for chunk in chunks if chunk.chunk_type == "text"),
        table_chunks=sum(1 for chunk in chunks if chunk.chunk_type == "table"),
        sections_processed=sections_processed,
        sections_skipped=sections_skipped,
        tables_processed=tables_processed,
        tables_skipped=tables_skipped,
    )

    result = ChunkingResult(
        document_id=document_id,
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
