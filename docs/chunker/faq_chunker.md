# FAQ Chunker

**Module:** `src/marley/chunker/faq_chunker.py`
**Input:** `data/knowledgebase/faq-stpo.json` (1039 entries), `data/knowledgebase/faq-ao.json`
**Output:** `data/chunks/faq-stpo-chunks.json`, `data/chunks/faq-ao-chunks.json`

The FAQ chunker converts FAQ knowledge bases into retrieval-ready chunks. Each question-answer entry becomes exactly one chunk — no sentence splitting, packing, or overlap is needed. Both FAQ sources (FAQ-StPO and FAQ-AO) share the same JSON structure and are handled by the same chunker.

---

## Theoretical Background

Frequently Asked Questions (FAQ) represent a form of structured knowledge where each entry is a self-contained question-answer pair. Unlike continuous prose, FAQ entries have a natural granularity that aligns directly with the retrieval unit: one question maps to one answer, and splitting an entry would break this relationship.

This motivates a **one-chunk-per-entry** design, which is the standard approach for FAQ-based knowledge bases in RAG systems. The chunker applies no sliding window, overlap, or merging — each valid entry becomes exactly one chunk in the format `"Question: {q}\nAnswer: {a}"`. This format preserves the question-answer structure and allows the retriever to match on both the question (useful for lexical similarity) and the answer (useful for semantic similarity).

The quality flag system monitors data integrity during chunking, detecting issues such as empty fields, duplicate IDs, and oversized entries. Entries that exceed the token budget are flagged but not split, since splitting would compromise the atomic question-answer relationship.

---

## Processing Pipeline

```
FAQ JSON file
  │
  load(path) → FAQDataset
  │
  chunk_faq(dataset):
  │
  For each entry:
  ├─ 1. Validate     (id, question, answer non-empty; no duplicates)
  ├─ 2. Format text   ("Question: {q}\nAnswer: {a}")
  ├─ 3. Count tokens  (tiktoken cl100k_base)
  ├─ 4. Build ID      ("faq-stpo-stpo-0001" / "faq-ao-ao-0001")
  ├─ 5. Build metadata
  └─ 6. Create FAQChunk
  │
  Compute stats, collect quality flags
  └── FAQChunkingResult → JSON
```

### Step 1: Validation

Each entry is validated for:

- Presence of a non-empty `id`, `question`, and `answer`
- No duplicate IDs within the same dataset

Invalid entries are skipped and recorded as quality flags.

### Step 2: Text Formatting

Question and answer are combined into a single retrieval-friendly text:

```
Question: What degree does the Computer Science program lead to?
Answer: The program leads to the degree of Master of Science (M.Sc.).
```

### Step 3: Token Counting

Tokens are counted using tiktoken with the `cl100k_base` encoding. Entries exceeding 512 tokens are flagged (info severity) but not split, since splitting would break the question-answer relationship.

### Step 4: Chunk ID

The chunk ID is built deterministically from the FAQ source and entry ID:

```
{faq_source}-{entry_id}
```

Examples: `faq-stpo-stpo-0001`, `faq-ao-ao-0001`

---

## Public API

```python
from src.marley.chunker import load_faq, chunk_faq, save_faq

dataset = load_faq("data/knowledgebase/faq-stpo.json")
result = chunk_faq(dataset, source_file="data/knowledgebase/faq-stpo.json")
save_faq(result, "data/chunks/faq-stpo-chunks.json")
```

| Function    | Signature                                                                                               | Description                                              |
| ----------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `load`      | `(faq_path: str \| Path) → FAQDataset`                                                                | Load a FAQ JSON file.                                    |
| `chunk_faq` | `(dataset: FAQDataset, source_file: str = "", *, tokenizer: str = "cl100k_base") → FAQChunkingResult` | Chunk a FAQ dataset. Each valid entry becomes one chunk. |
| `save`      | `(result: FAQChunkingResult, output_path: str \| Path) → Path`                                        | Serialize to JSON. Creates parent directories.           |

---

## Quality Flags

| Code                  | Severity | Trigger                              |
| --------------------- | -------- | ------------------------------------ |
| `FAQ_ENTRY_INVALID`   | warning  | Entry has no valid id                |
| `FAQ_ID_DUPLICATE`    | warning  | Duplicate ID in same dataset         |
| `FAQ_EMPTY_QUESTION`  | warning  | Question is empty or whitespace-only |
| `FAQ_EMPTY_ANSWER`    | warning  | Answer is empty or whitespace-only   |
| `FAQ_OVERSIZED_ENTRY` | info     | Combined Q+A exceeds 512 tokens      |
| `FAQ_ALL_SKIPPED`     | error    | No valid entries produced any chunks |

---

## Data Classes

The FAQ chunker defines `FAQEntry`, `FAQDataset`, `FAQChunkMetadata`, `FAQChunk`, `FAQChunkingStats`, and `FAQChunkingResult` locally. It imports `QualityFlag` and `compute_token_stats` from `src.marley.models/`. See `docs/models/models.md` for the shared data classes.

---

## Dependencies

| Library  | Purpose                                   |
| -------- | ----------------------------------------- |
| tiktoken | Token counting via `cl100k_base` encoding |

No sentence segmentation library is needed since FAQ entries are not split.

---

## Known Characteristics

- FAQ-StPO produces 1039 chunks (one per entry).
- FAQ-AO currently contains 1 placeholder entry with empty fields. The chunker correctly skips it and produces 0 chunks. Once real advisory office data is available, each valid entry will produce one chunk.
- Token counts range from 16 to 416, with typical values between 20 and 120 tokens per chunk.
- No entries in the current datasets exceed 512 tokens.
- Both FAQ sources share identical structure and processing logic.
