"""FAQ chunker for the MARley pipeline.

Converts FAQ knowledge bases (FAQ-StPO, FAQ-AO) into retrieval-ready
chunks. Each valid question-answer entry becomes exactly one chunk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import tiktoken

from src.marley.models import QualityFlag


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FAQEntry:
    """A single FAQ question-answer pair."""
    id: str
    question: str
    answer: str
    source: str


@dataclass
class FAQDataset:
    """A loaded FAQ knowledge base."""
    faq_source: str  # "faq-stpo" or "faq-ao"
    entries: list[FAQEntry]


@dataclass
class FAQChunkMetadata:
    """Metadata for a single FAQ chunk."""
    faq_source: str
    source_file: str
    faq_id: str
    source_reference: str
    chunk_index: int


@dataclass
class FAQChunk:
    """A single FAQ chunk (one per valid entry)."""
    chunk_id: str
    chunk_type: str  # always "faq"
    text: str
    token_count: int
    metadata: FAQChunkMetadata


@dataclass
class FAQChunkingStats:
    """Aggregated statistics over all FAQ chunks."""
    total_chunks: int
    entries_total: int
    entries_processed: int
    entries_skipped: int
    min_tokens: int
    median_tokens: int
    max_tokens: int
    total_tokens: int


@dataclass
class FAQChunkingResult:
    """Complete result of FAQ chunking."""
    faq_source: str
    source_file: str
    chunks: list[FAQChunk]
    stats: FAQChunkingStats
    quality_flags: list[QualityFlag]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_chunk_text(question: str, answer: str) -> str:
    """Combine question and answer into chunk text."""
    return f"Question: {question.strip()}\nAnswer: {answer.strip()}"


def _build_chunk_id(faq_source: str, faq_id: str) -> str:
    """Build a deterministic chunk ID from source and entry ID.

    Examples: "faq-stpo-0001", "faq-ao-0001".
    """
    return f"{faq_source}-{faq_id}"


def _validate_entry(
    raw: dict,
    index: int,
    seen_ids: set[str],
    flags: list[QualityFlag],
) -> FAQEntry | None:
    """Validate a raw FAQ entry dict and return a FAQEntry or None."""
    if not isinstance(raw, dict):
        flags.append(QualityFlag(
            code="FAQ_ENTRY_INVALID",
            message=f"Entry at index {index} is not a dict.",
            severity="warning",
            context={"index": index},
        ))
        return None

    entry_id = str(raw.get("id", "")).strip()
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    source = str(raw.get("source", "")).strip()

    if not entry_id:
        flags.append(QualityFlag(
            code="FAQ_ENTRY_INVALID",
            message=f"Entry at index {index} has no valid id.",
            severity="warning",
            context={"index": index},
        ))
        return None

    if entry_id in seen_ids:
        flags.append(QualityFlag(
            code="FAQ_ID_DUPLICATE",
            message=f"Duplicate FAQ id '{entry_id}' at index {index}.",
            severity="warning",
            context={"index": index, "faq_id": entry_id},
        ))
        return None

    if not question:
        flags.append(QualityFlag(
            code="FAQ_EMPTY_QUESTION",
            message=f"Entry '{entry_id}' has an empty question.",
            severity="warning",
            context={"faq_id": entry_id},
        ))
        return None

    if not answer:
        flags.append(QualityFlag(
            code="FAQ_EMPTY_ANSWER",
            message=f"Entry '{entry_id}' has an empty answer.",
            severity="warning",
            context={"faq_id": entry_id},
        ))
        return None

    seen_ids.add(entry_id)
    return FAQEntry(id=entry_id, question=question, answer=answer, source=source)


def _compute_stats(
    chunks: list[FAQChunk],
    entries_total: int,
    entries_skipped: int,
) -> FAQChunkingStats:
    """Compute aggregated statistics over all chunks."""
    token_counts = [c.token_count for c in chunks]

    if not token_counts:
        return FAQChunkingStats(
            total_chunks=0,
            entries_total=entries_total,
            entries_processed=0,
            entries_skipped=entries_skipped,
            min_tokens=0,
            median_tokens=0,
            max_tokens=0,
            total_tokens=0,
        )

    return FAQChunkingStats(
        total_chunks=len(token_counts),
        entries_total=entries_total,
        entries_processed=len(token_counts),
        entries_skipped=entries_skipped,
        min_tokens=min(token_counts),
        median_tokens=int(median(token_counts)),
        max_tokens=max(token_counts),
        total_tokens=sum(token_counts),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(faq_path: str | Path) -> FAQDataset:
    """Load a FAQ JSON file into a FAQDataset.

    Expects the JSON structure: {"metadata": {"source": "..."}, "entries": [...]}.
    """
    path = Path(faq_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    faq_source = data.get("metadata", {}).get("source", "")
    raw_entries = data.get("entries", [])

    entries: list[FAQEntry] = []
    for raw in raw_entries:
        if isinstance(raw, dict):
            entries.append(FAQEntry(
                id=str(raw.get("id", "")).strip(),
                question=str(raw.get("question", "")).strip(),
                answer=str(raw.get("answer", "")).strip(),
                source=str(raw.get("source", "")).strip(),
            ))

    return FAQDataset(faq_source=faq_source, entries=entries)


def chunk_faq(
    dataset: FAQDataset,
    source_file: str = "",
    *,
    tokenizer: str = "cl100k_base",
) -> FAQChunkingResult:
    """Chunk a FAQ dataset. Each valid entry becomes one chunk."""
    encoder = tiktoken.get_encoding(tokenizer)
    flags: list[QualityFlag] = []
    chunks: list[FAQChunk] = []
    seen_ids: set[str] = set()
    entries_skipped = 0

    for i, entry in enumerate(dataset.entries):
        raw = {"id": entry.id, "question": entry.question,
               "answer": entry.answer, "source": entry.source}
        validated = _validate_entry(raw, i, seen_ids, flags)
        if validated is None:
            entries_skipped += 1
            continue

        text = _format_chunk_text(validated.question, validated.answer)
        token_count = len(encoder.encode(text))
        chunk_id = _build_chunk_id(dataset.faq_source, validated.id)

        if token_count > 512:
            flags.append(QualityFlag(
                code="FAQ_OVERSIZED_ENTRY",
                message=f"Entry '{validated.id}' has {token_count} tokens (>512).",
                severity="info",
                context={"faq_id": validated.id, "token_count": token_count},
            ))

        chunks.append(FAQChunk(
            chunk_id=chunk_id,
            chunk_type="faq",
            text=text,
            token_count=token_count,
            metadata=FAQChunkMetadata(
                faq_source=dataset.faq_source,
                source_file=source_file,
                faq_id=validated.id,
                source_reference=validated.source,
                chunk_index=0,
            ),
        ))

    if not chunks and dataset.entries:
        flags.append(QualityFlag(
            code="FAQ_ALL_SKIPPED",
            message="All FAQ entries were skipped; no chunks produced.",
            severity="error",
        ))

    stats = _compute_stats(chunks, len(dataset.entries), entries_skipped)

    return FAQChunkingResult(
        faq_source=dataset.faq_source,
        source_file=source_file,
        chunks=chunks,
        stats=stats,
        quality_flags=flags,
    )


def save(result: FAQChunkingResult, output_path: str | Path) -> Path:
    """Save a FAQChunkingResult as JSON. Creates parent directories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path.resolve()
