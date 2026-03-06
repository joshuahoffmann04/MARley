# PDF Chunker

The PDF chunker splits the extracted StPO document into retrieval-ready chunks. It handles both plain text (sentence-aligned sliding windows) and tables (row-based packing with header repetition). It is the second stage of the MARley pipeline.

**Input:** `data/knowledgebase/stpo-extracted.json` (48 sections, 23 tables)
**Output:** `data/chunks/stpo-chunks.json`
**Module:** `src/marley/chunker/`

---

## Processing Pipeline

The chunker processes each section sequentially, producing text chunks and table chunks:

```
ExtractionResult
  │
  ├─ For each section:
  │   ├─ 1. Heading prefix        (build hierarchy path from parent)
  │   ├─ 2. Sentence splitting    (syntok, regex fallback)
  │   ├─ 3. Oversized splitting   (token-level for long sentences)
  │   ├─ 4. Greedy packing        (fill chunks up to max_tokens)
  │   ├─ 5. Undersized merging    (merge small chunks into neighbors)
  │   ├─ 6. Heading + overlap     (prepend path, add token overlap)
  │   │
  │   └─ For each table in section:
  │       ├─ 7. Row serialization  (pipe-delimited format)
  │       ├─ 8. Row packing        (fill chunks, repeat headers)
  │       └─ 9. Heading prefix     (same as text chunks)
  │
  └── ChunkingResult → JSON
```

### Stage 1: Heading Prefix

**Function:** `_build_heading_prefix(section, section_map) → (prefix, path_labels)`

Walks the section hierarchy via `parent_section_id` to build a breadcrumb path. For a paragraph, this produces a path like:

```
III. Examination-related provisions > §23 Master's Thesis
```

The prefix is prepended to every chunk from that section, providing retrieval context. Top-level sections without a parent (parts, appendices, preamble, ToC) include only their own label and title.

### Stage 2: Sentence Splitting

**Library:** syntok (preferred), regex fallback
**Function:** `_split_sentences(text) → list[str]`

Splits section text into individual sentences using syntok's segmenter. If syntok is unavailable, falls back to a regex split on sentence-ending punctuation (`[.!?]`). As a last resort, splits on newlines.

### Stage 3: Oversized Sentence Splitting

**Function:** `_split_oversized_sentence(sentence, encoder, max_tokens) → list[str]`

Sentences exceeding `max_chunk_tokens` are split at the token level using tiktoken. Each resulting piece is at most `max_tokens` tokens.

### Stage 4: Greedy Packing

**Function:** `_pack_sentences(sentences, encoder, max_tokens) → list[str]`

Sentences are packed into chunks greedily: each sentence is added to the current chunk if the combined token count stays within `max_tokens`. Otherwise, a new chunk begins. This ensures sentence boundaries are respected.

### Stage 5: Undersized Merging

**Function:** `_merge_undersized(chunks, encoder, min_tokens, max_tokens) → list[str]`

Chunks below `min_tokens` are merged into their forward neighbor if the combined size fits within `max_tokens`. If the forward merge would overflow, the chunk is merged backward into the previous chunk. This eliminates fragments that would be too small for meaningful retrieval.

### Stage 6: Heading and Overlap Application

**Function:** `_apply_heading_and_overlap(chunks, encoder, max_tokens, overlap_tokens, heading_prefix) → list[str]`

For each chunk:
1. The heading prefix is prepended (tokens are reserved from the budget).
2. For chunks after the first, a token-level overlap from the tail of the previous chunk is inserted before the current chunk's content. This provides continuity across chunk boundaries.

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
| `save` | `(result: ChunkingResult, output_path: str \| Path) → Path` | Serialize to JSON. Creates parent directories. Returns the resolved output path. |

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_chunk_tokens` | 512 | Maximum tokens per chunk. Matches typical embedding model context (nomic-embed-text). |
| `min_chunk_tokens` | 64 | Minimum tokens for a text chunk before merging into neighbors. |
| `overlap_tokens` | 50 | Number of tokens from the previous chunk to repeat as overlap (~10%). |
| `tokenizer` | `cl100k_base` | tiktoken encoding name. Used as a proxy for the Ollama embedding tokenizer. |

---

## Dependencies

| Library | Purpose |
|---|---|
| syntok | Sentence segmentation (preferred) |
| tiktoken | Token counting and encoding via `cl100k_base` |

Both are pure Python packages. The chunker also imports `ExtractionResult`, `Section`, and `Table` from `src.marley.extractor`.

---

## Known Characteristics

- All section kinds are chunked (preamble, ToC, parts, paragraphs, appendices).
- The preamble and some short paragraphs produce a single chunk each.
- Appendix 2's module table (54 rows, 7 columns) is split into multiple table chunks with repeated headers.
- Appendix 3's 14 separate tables each produce independent table chunks.
- Quality flags are collected but no error-level flags are expected in normal operation.
