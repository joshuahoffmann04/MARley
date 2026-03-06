# PDF Chunker

**Module:** `src/marley/chunker/pdf_chunker.py`
**Input:** `data/knowledgebase/stpo-extracted.json` (48 sections, 23 tables)
**Output:** `data/chunks/stpo-chunks.json`

The PDF chunker splits the extracted StPO document into retrieval-ready chunks. It uses a sentence-aligned sliding window for text and row-based packing with header repetition for tables. It is the second stage of the MARley pipeline.

---

## Processing Pipeline

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

### Stage 1: Heading Prefix

**Function:** `_build_heading_prefix(section, section_map) → (prefix, path_labels)`

Walks the section hierarchy via `parent_section_id` to build a breadcrumb path. For a paragraph, this produces a path like:

```
III. Examination-related provisions > §23 Master's Thesis
```

The prefix is prepended to every chunk from that section, providing retrieval context.

### Stage 2: Sentence Splitting

**Library:** syntok (preferred), regex fallback
**Function:** `_split_sentences(text) → list[str]`

Splits section text into individual sentences using syntok's segmenter. If syntok is unavailable, falls back to a regex split on sentence-ending punctuation (`[.!?]`). As a last resort, splits on newlines.

### Stage 3: Oversized Sentence Splitting

**Function:** `_split_oversized_sentence(sentence, encoder, max_tokens) → list[str]`

Sentences exceeding the token budget are split at the token level using tiktoken. Each resulting piece is at most `max_tokens` tokens.

**Function:** `_prepare_sentences(sentences, encoder, max_tokens) → (flat_sentences, token_counts)`

Applies oversized splitting to all sentences and pre-computes per-sentence token counts for efficient window construction.

### Stage 4: Sliding Window

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

### Stage 5: Undersized Merging

**Function:** `_merge_undersized(chunks, encoder, min_tokens, max_tokens) → list[str]`

Chunks below `min_tokens` are merged into their forward neighbour if the combined size fits within `max_tokens`. If the forward merge would overflow, the chunk is merged backward into the previous chunk. This eliminates fragments that would be too small for meaningful retrieval.

### Stage 6: Heading Application

**Function:** `_apply_heading_prefix(chunks, encoder, max_tokens, heading_prefix) → list[str]`

Prepends the heading prefix to every chunk. The heading token count is reserved from the budget before the sliding window runs, so the combined heading + body never exceeds `max_tokens`.

### Stages 7–9: Table Chunking

**Function:** `_build_table_chunks(table, encoder, max_tokens, heading_prefix) → list[str]`

Each table is chunked by rows:
1. Rows are serialized as pipe-delimited strings (`col1 | col2 | col3`).
2. Rows are packed into chunks until `max_tokens` is reached.
3. The header line is repeated at the top of each chunk so every chunk is self-contained.
4. The same heading prefix used for text chunks is prepended.

---

## Public API

```python
from src.marley.chunker import chunk_stpo, save

result = chunk_stpo(extraction_result)
save(result, "data/chunks/stpo-chunks.json")
```

| Function | Signature | Description |
|---|---|---|
| `chunk_stpo` | `(extraction: ExtractionResult, *, max_chunk_tokens=512, min_chunk_tokens=64, overlap_tokens=50, tokenizer="cl100k_base") → ChunkingResult` | Chunk an extracted StPO document. |
| `save` | `(result: ChunkingResult, output_path: str \| Path) → Path` | Serialize to JSON. Delegates to `save_json` from `src.marley.models`. |

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_tokens` | 512 | Maximum tokens per chunk (heading + body). |
| `min_chunk_tokens` | 64 | Minimum tokens for a text chunk before merging into neighbours. |
| `overlap_tokens` | 50 | Target token count for sentence-aligned overlap between consecutive windows (~10%). |
| `tokenizer` | `cl100k_base` | tiktoken encoding name. Used as a proxy for the embedding model tokenizer. |

---

## Data Classes

The chunker defines `ChunkMetadata`, `Chunk`, `ChunkingStats`, and `ChunkingResult` locally. It imports `ExtractionResult`, `Section`, `Table`, `QualityFlag`, `compute_token_stats`, and `save_json` from `src.marley.models/`. See `docs/models/models.md` for the shared data classes.

---

## Dependencies

| Library | Purpose |
|---|---|
| syntok | Sentence segmentation (preferred) |
| tiktoken | Token counting and encoding via `cl100k_base` |

---

## Known Characteristics

- All section kinds are chunked (preamble, ToC, parts, paragraphs, appendices).
- The preamble and some short paragraphs produce a single chunk each.
- Sections with multiple text chunks share sentence-aligned overlap (~50 tokens).
- When a single sentence fills the entire token budget, overlap with the adjacent chunk is not possible.
- Appendix 2's module table (54 rows, 7 columns) is split into multiple table chunks with repeated headers.
- Appendix 3's 14 separate tables each produce independent table chunks.
- Quality flags are collected but no error-level flags are expected in normal operation.
