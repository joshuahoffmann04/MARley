# Chunking Pipeline

**Module:** `src/marley/chunker/`

The chunker module splits extracted documents into retrieval-ready chunks. Two chunkers address different input formats:

1. **PDF Chunker** (`pdf_chunker.py`): Sentence-aligned sliding window for extracted StPO sections and row-based packing for tables.
2. **FAQ Chunker** (`faq_chunker.py`): One chunk per question-answer pair from FAQ knowledge bases.

---

## Module Structure

```
src/marley/chunker/
├── __init__.py          # Re-exports all public symbols
├── pdf_chunker.py       # StPO text + table chunking
└── faq_chunker.py       # FAQ Q&A chunking
```

---

## Part 1: PDF Chunker

**Input:** `data/knowledgebase/stpo-extracted.json` (48 sections, 23 tables)
**Output:** `data/chunks/stpo-chunks.json`

The PDF chunker splits the extracted StPO document into retrieval-ready chunks. It uses a sentence-aligned sliding window for text and row-based packing with header repetition for tables.

### Theoretical Background

Chunking is a critical preprocessing step in Retrieval-Augmented Generation (RAG) pipelines. The retriever operates on chunks, not full documents, so the chunking strategy directly affects retrieval precision and recall. Three common strategies exist in practice:

1. **Fixed-size chunking** splits text at a fixed token count. It is simple but frequently cuts mid-sentence, producing chunks with incomplete semantic units.
2. **Recursive/separator-based chunking** splits along structural boundaries (paragraphs, headings). It preserves document structure but produces chunks of highly variable size, which can exceed embedding model limits.
3. **Sliding-window chunking** moves a fixed-size window across the text with overlap between consecutive chunks. It guarantees size constraints while providing redundancy at boundaries.

MARley uses a **sentence-aligned sliding window** — a variant of (3) where window boundaries are snapped to sentence boundaries. This combines the size guarantees of fixed-window chunking with the semantic coherence of sentence-level splitting. The overlap region always consists of complete sentences, so no information is lost at chunk boundaries.

The overlap of approximately 50 tokens (~10% of the 512-token budget) follows the practical guideline that 10–20% overlap provides sufficient boundary coverage without excessive redundancy (LangChain documentation; LlamaIndex best practices). Token counting uses the `cl100k_base` tokenizer as a proxy for the embedding model's tokenizer.

For tables, a row-based packing strategy is used: rows are serialized as pipe-delimited strings, packed into chunks until the token budget is reached, and each chunk receives a repeated header line. This ensures every table chunk is self-contained and interpretable by the retriever and generator.

### Processing Pipeline

```
ExtractionResult
  │
  ├─ For each section:
  │   ├─ 1. Heading prefix        Build hierarchy path from parent
  │   ├─ 2. Sentence splitting    syntok (preferred), regex fallback
  │   ├─ 3. Oversized splitting   Token-level split for long sentences
  │   ├─ 4. Sliding window        Sentence-aligned windows with overlap
  │   ├─ 5. Undersized merging    Merge small chunks into neighbours
  │   ├─ 6. Heading application   Prepend section path to each chunk
  │   │
  │   └─ For each table in section:
  │       ├─ 7. Row serialization  Pipe-delimited format
  │       ├─ 8. Row packing        Fill chunks, repeat headers
  │       └─ 9. Heading prefix     Same as text chunks
  │
  └── ChunkingResult → JSON
```

#### Stage 1: Heading Prefix

**Function:** `_build_heading_prefix(section, section_map) → (prefix, path_labels)`

Walks the section hierarchy via `parent_section_id` to build a breadcrumb path. For a paragraph, this produces a path like:

```
III. Examination-related provisions > §23 Master's Thesis
```

The prefix is prepended to every chunk from that section, providing retrieval context.

#### Stage 2: Sentence Splitting

**Library:** syntok (preferred), regex fallback
**Function:** `_split_sentences(text) → list[str]`

Splits section text into individual sentences using syntok's segmenter. If syntok is unavailable, falls back to a regex split on sentence-ending punctuation (`[.!?]`). As a last resort, splits on newlines.

#### Stage 3: Oversized Sentence Splitting

**Function:** `_split_oversized_sentence(sentence, encoder, max_tokens) → list[str]`

Sentences exceeding the token budget are split at the token level using tiktoken. Each resulting piece is at most `max_tokens` tokens.

**Function:** `_prepare_sentences(sentences, encoder, max_tokens) → (flat_sentences, token_counts)`

Applies oversized splitting to all sentences and pre-computes per-sentence token counts for efficient window construction.

#### Stage 4: Sliding Window

**Function:** `_sliding_window_chunks(sentences, token_counts, max_tokens, overlap_tokens) → list[str]`

Builds chunks by sliding a window over the sentence list:

1. **Expand:** Starting from the current position, add sentences until the next sentence would exceed the token budget.
2. **Record:** The current window becomes a chunk.
3. **Slide:** Advance the start position so that approximately `overlap_tokens` worth of trailing sentences are shared with the next window.

This produces sentence-aligned overlap: the shared content between consecutive chunks always consists of complete sentences. This is preferable to token-level overlap (which can cut mid-sentence) because it preserves semantic coherence at chunk boundaries.

**Overlap example** (overlap_tokens=50):
```
Chunk 1: [S1  S2  S3  S4  S5]
                    ─────────── ~50 tokens overlap
Chunk 2:           [S4  S5  S6  S7  S8]
                              ────────── ~50 tokens overlap
Chunk 3:                     [S7  S8  S9  S10]
```

**Edge case:** When a single sentence fills the entire budget, no overlap with the next chunk is possible. The algorithm guarantees forward progress (at least one sentence per iteration).

#### Stage 5: Undersized Merging

**Function:** `_merge_undersized(chunks, encoder, min_tokens, max_tokens) → list[str]`

Chunks below `min_tokens` are merged into their forward neighbour if the combined size fits within `max_tokens`. If the forward merge would overflow, the chunk is merged backward into the previous chunk. This eliminates fragments that would be too small for meaningful retrieval.

#### Stage 6: Heading Application

**Function:** `_apply_heading_prefix(chunks, encoder, max_tokens, heading_prefix) → list[str]`

Prepends the heading prefix to every chunk. The heading token count is reserved from the budget before the sliding window runs, so the combined heading + body never exceeds `max_tokens`.

#### Stages 7–9: Table Chunking

**Function:** `_build_table_chunks(table, encoder, max_tokens, heading_prefix) → list[str]`

Each table is chunked by rows:
1. Rows are serialized as pipe-delimited strings (`col1 | col2 | col3`).
2. Rows are packed into chunks until `max_tokens` is reached.
3. The header line is repeated at the top of each chunk so every chunk is self-contained.
4. The same heading prefix used for text chunks is prepended.

### Public API

```python
from src.marley.chunker import chunk_stpo, save

result = chunk_stpo(extraction_result)
save(result, "data/chunks/stpo-chunks.json")
```

| Function | Signature | Description |
|---|---|---|
| `chunk_stpo` | `(extraction: ExtractionResult, *, max_chunk_tokens=512, min_chunk_tokens=64, overlap_tokens=50, tokenizer="cl100k_base") → ChunkingResult` | Chunk an extracted StPO document. |
| `save` | `(result: ChunkingResult, output_path: str \| Path) → Path` | Serialize to JSON. Delegates to `save_json` from `src.marley.models`. |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_tokens` | 512 | Maximum tokens per chunk (heading + body). |
| `min_chunk_tokens` | 64 | Minimum tokens for a text chunk before merging into neighbours. |
| `overlap_tokens` | 50 | Target token count for sentence-aligned overlap between consecutive windows (~10%). |
| `tokenizer` | `cl100k_base` | tiktoken encoding name. Used as a proxy for the embedding model tokenizer. |

### Data Classes

#### ChunkMetadata

```python
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
```

#### Chunk

```python
@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str       # "text" or "table"
    text: str
    token_count: int
    metadata: ChunkMetadata
```

#### ChunkingStats

```python
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
```

#### ChunkingResult

```python
@dataclass
class ChunkingResult:
    source_file: str
    chunks: list[Chunk]
    stats: ChunkingStats
    quality_flags: list[QualityFlag]
```

---

## Part 2: FAQ Chunker

**Input:** `data/knowledgebase/faq-stpo.json` or `data/knowledgebase/faq-ao.json`
**Output:** `data/chunks/faq-stpo-chunks.json` or `data/chunks/faq-ao-chunks.json`

The FAQ chunker converts FAQ knowledge bases into retrieval-ready chunks. Each valid question-answer entry becomes exactly one chunk. No sliding window, no overlap — each Q&A pair is a self-contained retrieval unit.

### Processing

1. **Load** the FAQ JSON file via `load()` into a `FAQDataset`.
2. **Validate** each entry: must have a non-empty `id` (unique), `question`, and `answer`. Invalid entries are skipped and recorded as quality flags.
3. **Format** each valid entry as `"Question: {question}\nAnswer: {answer}"`.
4. **Count tokens** using tiktoken (`cl100k_base`).
5. **Build chunk ID** as `"{faq_source}-{entry_id}"` (e.g., `"faq-stpo-stpo-0001"`).
6. Entries exceeding `DEFAULT_MAX_CHUNK_TOKENS` (512) receive an info-level quality flag but are kept.

### Public API

```python
from src.marley.chunker import load_faq, chunk_faq, save_faq

dataset = load_faq("data/knowledgebase/faq-stpo.json")
result = chunk_faq(dataset, source_file="data/knowledgebase/faq-stpo.json")
save_faq(result, "data/chunks/faq-stpo-chunks.json")
```

| Function | Signature | Description |
|---|---|---|
| `load` | `(faq_path: str \| Path) → FAQDataset` | Load a FAQ JSON file into a FAQDataset. Imported as `load_faq` from the package. |
| `chunk_faq` | `(dataset: FAQDataset, source_file: str = "", *, tokenizer: str = "cl100k_base") → FAQChunkingResult` | Chunk a FAQ dataset. Each valid entry becomes one chunk. |
| `save` | `(result: FAQChunkingResult, output_path: str \| Path) → Path` | Serialize to JSON. Imported as `save_faq` from the package. |

### Validation Rules

| Code | Severity | Condition |
|---|---|---|
| `FAQ_ENTRY_INVALID` | warning | Entry has no valid id. |
| `FAQ_ID_DUPLICATE` | warning | Duplicate FAQ id. |
| `FAQ_EMPTY_QUESTION` | warning | Entry has an empty question. |
| `FAQ_EMPTY_ANSWER` | warning | Entry has an empty answer. |
| `FAQ_OVERSIZED_ENTRY` | info | Entry exceeds 512 tokens. |
| `FAQ_ALL_SKIPPED` | error | All entries were skipped; no chunks produced. |

### Data Classes

#### FAQEntry

```python
@dataclass
class FAQEntry:
    id: str
    question: str
    answer: str
    source: str
```

#### FAQDataset

```python
@dataclass
class FAQDataset:
    faq_source: str         # "faq-stpo" or "faq-ao"
    entries: list[FAQEntry]
```

#### FAQChunkMetadata

```python
@dataclass
class FAQChunkMetadata:
    faq_source: str
    source_file: str
    faq_id: str
    source_reference: str
    chunk_index: int        # Always 0 (one chunk per entry)
```

#### FAQChunk

```python
@dataclass
class FAQChunk:
    chunk_id: str
    chunk_type: str         # Always "faq"
    text: str
    token_count: int
    metadata: FAQChunkMetadata
```

#### FAQChunkingStats

```python
@dataclass
class FAQChunkingStats:
    total_chunks: int
    entries_total: int
    entries_processed: int
    entries_skipped: int
    min_tokens: int
    median_tokens: int
    max_tokens: int
    total_tokens: int
```

#### FAQChunkingResult

```python
@dataclass
class FAQChunkingResult:
    faq_source: str
    source_file: str
    chunks: list[FAQChunk]
    stats: FAQChunkingStats
    quality_flags: list[QualityFlag]
```

---

## Dependencies

| Library | Purpose |
|---|---|
| syntok | Sentence segmentation (preferred, PDF chunker only) |
| tiktoken | Token counting and encoding via `cl100k_base` |

---

## Imports

All public symbols are available from the package root:

```python
from src.marley.chunker import (
    # PDF Chunker
    Chunk, ChunkingResult, ChunkingStats, ChunkMetadata,
    chunk_stpo, save,
    # FAQ Chunker
    FAQChunk, FAQChunkingResult, FAQChunkingStats, FAQChunkMetadata,
    FAQDataset, FAQEntry,
    chunk_faq, load_faq, save_faq,
    # Shared
    QualityFlag,
)
```

---

## Known Characteristics

- All section kinds are chunked (preamble, ToC, parts, paragraphs, appendices).
- The preamble and some short paragraphs produce a single chunk each.
- Sections with multiple text chunks share sentence-aligned overlap (~50 tokens).
- When a single sentence fills the entire token budget, overlap with the adjacent chunk is not possible.
- Appendix 2's module table (54 rows, 7 columns) is split into multiple table chunks with repeated headers.
- Appendix 3's 14 separate tables each produce independent table chunks.
- Quality flags are collected but no error-level flags are expected in normal operation.
- FAQ entries are never split — each Q&A pair is one chunk regardless of token count.
