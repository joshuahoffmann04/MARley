"""PDF chunker for the StPO document.

Splits extracted StPO sections into retrieval-ready chunks using
sentence-aligned sliding windows for text and row-based packing
with header repetition for tables.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import tiktoken

from src.marley.models import ExtractionResult, QualityFlag, Section, Table

try:
    import syntok.segmenter as _syntok_segmenter
except ImportError:
    _syntok_segmenter = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    document_id: str
    source_file: str
    section_id: str | None
    section_kind: str | None
    section_label: str | None
    section_title: str | None
    parent_section_id: str | None
    heading_path: list[str]
    start_page: int | None
    end_page: int | None
    chunk_index: int
    table_id: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str  # "text" or "table"
    text: str
    token_count: int
    metadata: ChunkMetadata


@dataclass
class ChunkingStats:
    total_chunks: int
    text_chunks: int
    table_chunks: int
    sections_processed: int
    sections_skipped: int
    tables_processed: int
    min_tokens: int
    median_tokens: int
    max_tokens: int
    total_tokens: int


@dataclass
class ChunkingResult:
    source_file: str
    chunks: list[Chunk]
    stats: ChunkingStats
    quality_flags: list[QualityFlag]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _join_sentence_tokens(sentence) -> str:
    return "".join(token.value + token.spacing for token in sentence).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using syntok with a regex fallback."""
    if not text.strip():
        return []

    if _syntok_segmenter is not None:
        sentences: list[str] = []
        for paragraph in _syntok_segmenter.process(text):
            for sentence in paragraph:
                candidate = _join_sentence_tokens(sentence)
                if candidate:
                    sentences.append(candidate)
        if sentences:
            return sentences

    rough = re.split(r"(?<=[.!?])\s+", text)
    results = [part.strip() for part in rough if part.strip()]
    if results:
        return results

    return [line.strip() for line in text.splitlines() if line.strip()] or [text.strip()]


def _split_oversized_sentence(
    sentence: str,
    encoder: tiktoken.Encoding,
    max_tokens: int,
) -> list[str]:
    """Split a sentence that exceeds max_tokens at the token level."""
    token_ids = encoder.encode(sentence)
    if len(token_ids) <= max_tokens:
        return [sentence]

    parts: list[str] = []
    for start in range(0, len(token_ids), max_tokens):
        piece = encoder.decode(token_ids[start:start + max_tokens]).strip()
        if piece:
            parts.append(piece)
    return parts


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _pack_sentences(
    sentences: list[str],
    encoder: tiktoken.Encoding,
    max_tokens: int,
) -> list[str]:
    """Greedily pack sentences into chunks respecting max_tokens."""
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        parts = _split_oversized_sentence(sentence, encoder, max_tokens)
        for piece in parts:
            if not current:
                current = [piece]
                continue
            candidate = " ".join([*current, piece])
            if len(encoder.encode(candidate)) <= max_tokens:
                current.append(piece)
            else:
                chunks.append(" ".join(current))
                current = [piece]

    if current:
        chunks.append(" ".join(current))

    return chunks


def _merge_undersized(
    chunks: list[str],
    encoder: tiktoken.Encoding,
    min_tokens: int,
    max_tokens: int,
) -> list[str]:
    """Merge chunks below min_tokens into their neighbors."""
    if len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        current_tokens = len(encoder.encode(current))

        if current_tokens >= min_tokens:
            merged.append(current)
            i += 1
            continue

        if i + 1 < len(chunks):
            candidate = f"{current} {chunks[i + 1]}"
            if len(encoder.encode(candidate)) <= max_tokens:
                chunks[i + 1] = candidate
                i += 1
                continue

        if merged:
            candidate = f"{merged[-1]} {current}"
            if len(encoder.encode(candidate)) <= max_tokens:
                merged[-1] = candidate
                i += 1
                continue

        merged.append(current)
        i += 1

    return merged


def _build_heading_prefix(
    section: Section,
    section_map: dict[str, Section],
) -> tuple[str | None, list[str]]:
    """Build a heading path prefix from the section hierarchy.

    Returns (prefix_string, path_labels) where prefix_string is the
    formatted heading to prepend, and path_labels is the list of
    label-title pairs for metadata.
    """
    path: list[Section] = []
    current: Section | None = section
    guard = 0
    while current is not None and guard < 10:
        path.append(current)
        parent_id = current.parent_section_id
        current = section_map.get(parent_id) if parent_id else None
        guard += 1
    path.reverse()

    labels: list[str] = []
    for node in path:
        label = (node.label or "").strip()
        title = (node.title or "").strip()
        if label and title:
            labels.append(f"{label} {title}")
        elif label:
            labels.append(label)
        elif title:
            labels.append(title)

    if not labels:
        return None, []

    prefix = " > ".join(labels) + "\n\n"
    return prefix, labels


def _apply_heading_and_overlap(
    chunks: list[str],
    encoder: tiktoken.Encoding,
    max_tokens: int,
    overlap_tokens: int,
    heading_prefix: str | None,
) -> list[str]:
    """Prepend heading prefix and add token overlap between chunks."""
    if not chunks:
        return []

    heading_token_ids = encoder.encode(heading_prefix) if heading_prefix else []
    body_budget = max_tokens - len(heading_token_ids)
    if body_budget <= 0:
        body_budget = max_tokens
        heading_token_ids = []

    result: list[str] = []
    prev_token_ids: list[int] = []

    for i, chunk in enumerate(chunks):
        current_token_ids = encoder.encode(chunk)
        if len(current_token_ids) > body_budget:
            current_token_ids = current_token_ids[:body_budget]

        overlap_ids: list[int] = []
        if i > 0 and overlap_tokens > 0 and prev_token_ids:
            capacity = body_budget - len(current_token_ids)
            n_overlap = min(overlap_tokens, max(capacity, 0))
            if n_overlap > 0:
                overlap_ids = prev_token_ids[-n_overlap:]

        body_ids = [*overlap_ids, *current_token_ids]
        if len(body_ids) > body_budget:
            body_ids = body_ids[:body_budget]

        final_ids = [*heading_token_ids, *body_ids] if heading_token_ids else body_ids
        final_text = encoder.decode(final_ids).strip()
        if final_text:
            result.append(final_text)

        prev_token_ids = current_token_ids

    return result


def _chunk_section_text(
    section: Section,
    section_map: dict[str, Section],
    encoder: tiktoken.Encoding,
    max_chunk_tokens: int,
    min_chunk_tokens: int,
    overlap_tokens: int,
    quality_flags: list[QualityFlag],
) -> tuple[list[str], str | None, list[str]]:
    """Chunk a section's text content.

    Returns (chunk_texts, heading_prefix, path_labels).
    """
    heading_prefix, path_labels = _build_heading_prefix(section, section_map)

    text = section.text.strip()
    if not text:
        return [], heading_prefix, path_labels

    sentences = _split_sentences(text)
    packed = _pack_sentences(sentences, encoder, max_chunk_tokens)
    merged = _merge_undersized(packed, encoder, min_chunk_tokens, max_chunk_tokens)
    final = _apply_heading_and_overlap(
        merged, encoder, max_chunk_tokens, overlap_tokens, heading_prefix,
    )
    return final, heading_prefix, path_labels


# ---------------------------------------------------------------------------
# Table chunking
# ---------------------------------------------------------------------------

def _serialize_table_row(row: list[str]) -> str:
    """Serialize a table row as pipe-delimited text."""
    return " | ".join(cell.strip() for cell in row if cell and cell.strip())


def _build_table_chunks(
    table: Table,
    encoder: tiktoken.Encoding,
    max_tokens: int,
    heading_prefix: str | None,
) -> list[str]:
    """Pack table rows into chunks, repeating headers in each chunk."""
    if not table.rows:
        return []

    heading_token_ids = encoder.encode(heading_prefix) if heading_prefix else []
    body_budget = max_tokens - len(heading_token_ids)
    if body_budget <= 0:
        body_budget = max_tokens
        heading_token_ids = []

    header_line = " | ".join(h.strip() for h in table.headers if h and h.strip())
    header_tokens = len(encoder.encode(header_line + "\n")) if header_line else 0
    row_budget = body_budget - header_tokens

    if row_budget <= 0:
        row_budget = body_budget
        header_line = ""
        header_tokens = 0

    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    for row in table.rows:
        row_line = _serialize_table_row(row)
        if not row_line:
            continue
        row_line_tokens = len(encoder.encode(row_line))

        if row_line_tokens > row_budget:
            if current_lines:
                body = header_line + "\n" + "\n".join(current_lines) if header_line else "\n".join(current_lines)
                chunks.append(body)
                current_lines = []
                current_tokens = 0
            chunks.append(row_line)
            continue

        if current_tokens + row_line_tokens + 1 > row_budget and current_lines:
            body = header_line + "\n" + "\n".join(current_lines) if header_line else "\n".join(current_lines)
            chunks.append(body)
            current_lines = []
            current_tokens = 0

        current_lines.append(row_line)
        current_tokens += row_line_tokens + 1

    if current_lines:
        body = header_line + "\n" + "\n".join(current_lines) if header_line else "\n".join(current_lines)
        chunks.append(body)

    if not heading_token_ids:
        return [c for c in chunks if c.strip()]

    prefixed: list[str] = []
    for chunk in chunks:
        body_ids = encoder.encode(chunk)[:body_budget]
        final_text = encoder.decode([*heading_token_ids, *body_ids]).strip()
        if final_text:
            prefixed.append(final_text)
    return prefixed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    chunks: list[Chunk],
    sections_processed: int,
    sections_skipped: int,
    tables_processed: int,
) -> ChunkingStats:
    token_counts = [c.token_count for c in chunks]
    text_count = sum(1 for c in chunks if c.chunk_type == "text")
    table_count = sum(1 for c in chunks if c.chunk_type == "table")

    if not token_counts:
        return ChunkingStats(
            total_chunks=0, text_chunks=0, table_chunks=0,
            sections_processed=sections_processed,
            sections_skipped=sections_skipped,
            tables_processed=tables_processed,
            min_tokens=0, median_tokens=0, max_tokens=0, total_tokens=0,
        )

    return ChunkingStats(
        total_chunks=len(token_counts),
        text_chunks=text_count,
        table_chunks=table_count,
        sections_processed=sections_processed,
        sections_skipped=sections_skipped,
        tables_processed=tables_processed,
        min_tokens=min(token_counts),
        median_tokens=int(median(token_counts)),
        max_tokens=max(token_counts),
        total_tokens=sum(token_counts),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_stpo(
    extraction: ExtractionResult,
    *,
    max_chunk_tokens: int = 512,
    min_chunk_tokens: int = 64,
    overlap_tokens: int = 50,
    tokenizer: str = "cl100k_base",
) -> ChunkingResult:
    """Chunk an extracted StPO document into retrieval-ready pieces."""
    encoder = tiktoken.get_encoding(tokenizer)
    section_map = {s.section_id: s for s in extraction.sections}
    quality_flags: list[QualityFlag] = []
    chunks: list[Chunk] = []
    sections_processed = 0
    sections_skipped = 0
    tables_processed = 0

    if _syntok_segmenter is None:
        quality_flags.append(QualityFlag(
            code="SYNTOK_UNAVAILABLE",
            message="syntok is not installed; regex sentence splitting is active.",
            severity="warning",
        ))

    for section in extraction.sections:
        text = section.text.strip()
        if not text and not section.tables:
            sections_skipped += 1
            quality_flags.append(QualityFlag(
                code="SECTION_EMPTY",
                message=f"Section {section.section_id} has no text and no tables.",
                severity="info",
                context={"section_id": section.section_id},
            ))
            continue

        sections_processed += 1
        chunk_texts, heading_prefix, path_labels = _chunk_section_text(
            section, section_map, encoder,
            max_chunk_tokens, min_chunk_tokens, overlap_tokens,
            quality_flags,
        )

        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{section.section_id}-txt-{i + 1}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                chunk_type="text",
                text=chunk_text,
                token_count=len(encoder.encode(chunk_text)),
                metadata=ChunkMetadata(
                    document_id=extraction.source_file,
                    source_file=extraction.source_file,
                    section_id=section.section_id,
                    section_kind=section.kind,
                    section_label=section.label,
                    section_title=section.title,
                    parent_section_id=section.parent_section_id,
                    heading_path=path_labels,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    chunk_index=i,
                ),
            ))

        for table in section.tables:
            table_chunks = _build_table_chunks(
                table, encoder, max_chunk_tokens, heading_prefix,
            )
            if not table_chunks:
                continue

            tables_processed += 1
            for j, chunk_text in enumerate(table_chunks):
                chunk_id = f"{section.section_id}-tbl-{table.table_id}-{j + 1}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    chunk_type="table",
                    text=chunk_text,
                    token_count=len(encoder.encode(chunk_text)),
                    metadata=ChunkMetadata(
                        document_id=extraction.source_file,
                        source_file=extraction.source_file,
                        section_id=section.section_id,
                        section_kind=section.kind,
                        section_label=section.label,
                        section_title=section.title,
                        parent_section_id=section.parent_section_id,
                        heading_path=path_labels,
                        start_page=section.start_page,
                        end_page=section.end_page,
                        chunk_index=len(chunk_texts) + j,
                        table_id=table.table_id,
                    ),
                ))

    stats = _compute_stats(chunks, sections_processed, sections_skipped, tables_processed)

    return ChunkingResult(
        source_file=extraction.source_file,
        chunks=chunks,
        stats=stats,
        quality_flags=quality_flags,
    )


def save(result: ChunkingResult, output_path: str | Path) -> Path:
    """Save a ChunkingResult as JSON. Creates parent directories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path.resolve()
