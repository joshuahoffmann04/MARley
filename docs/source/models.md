# Data Models

**Module:** `src/marley/models/`

The models package defines shared data classes, abstract base classes, utility functions, and constants used across all pipeline stages. It serves as the single source of truth for data contracts between extractor, chunker, retrieval, generation, abstention, and evaluation components.

---

## Module Structure

```
src/marley/models/
├── __init__.py       # Re-exports all public symbols
├── constants.py      # Pipeline-wide default values and enumerations
├── extraction.py     # ExtractionResult, Section, Table
├── quality.py        # QualityFlag
├── retrieval.py      # Retriever (ABC), RetrievalResult, rrf_fuse, load_chunks, validate_corpus
├── generation.py     # Generator (ABC), GenerationResult
├── abstention.py     # AbstentionResult
├── scoring.py        # normalize_scores, filter_by_threshold, compute_confidence
├── chunking.py       # compute_token_stats
└── io.py             # save_json
```

---

## Extraction

### ExtractionResult

Complete output of the PDF extractor. Represents a full document decomposed into hierarchical sections.

```python
@dataclass
class ExtractionResult:
    source_file: str         # Path to the source PDF
    total_pages: int         # Number of pages in the PDF
    sections: list[Section]  # Ordered list of document sections
```

### Section

A structural unit of the StPO document (paragraph, part, appendix, etc.).

```python
@dataclass
class Section:
    section_id: str                       # Unique ID (e.g., "par-23", "appendix-2")
    label: str                            # Display label (e.g., "§23")
    title: str                            # Section title
    kind: str                             # "preamble", "toc", "part", "paragraph", "appendix"
    start_page: int                       # First page (1-indexed)
    end_page: int                         # Last page (1-indexed)
    text: str                             # Full plain text
    tables: list[Table] = []              # Tables on this section's pages
    parent_section_id: str | None = None  # Parent part ID for paragraphs
```

### Table

A table extracted from a PDF page.

```python
@dataclass
class Table:
    table_id: str          # Unique ID (e.g., "par-6-tbl-1")
    page: int              # Page where the table starts
    headers: list[str]     # Column header strings
    rows: list[list[str]]  # Data rows
```

---

## Retrieval

### Retriever

Abstract base class for all retrieval strategies. Concrete implementations (`BM25Retriever`, `VectorRetriever`, `HybridRetriever`, `FusionRetriever`, `MergedRetriever`) are defined in `src/marley/retrieval/`.

```python
class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[RetrievalResult]: ...

    @property
    @abstractmethod
    def size(self) -> int: ...
```

| Method | Description |
|---|---|
| `index(corpus)` | Build the retrieval index from a list of chunk dicts (keys: `chunk_id`, `text`, `metadata`). |
| `retrieve(query, k)` | Return the top-*k* most relevant chunks, sorted by descending score. |
| `size` | Number of indexed chunks (read-only property). |

### RetrievalResult

A single retrieval hit with its relevance score.

```python
@dataclass
class RetrievalResult:
    chunk_id: str              # Chunk identifier
    text: str                  # Chunk text content
    score: float               # Relevance score (raw or normalized)
    metadata: dict[str, Any]   # Source metadata (section_id, label, etc.)
```

### Corpus Utilities

Functions for loading and validating the chunk corpus used by all retrievers.

#### `load_chunks(chunk_path)`

Load chunks from a JSON file produced by the chunking pipeline. Returns a list of dicts with `chunk_id`, `text`, and `metadata` keys.

```python
from src.marley.models import load_chunks

chunks = load_chunks("data/chunks/stpo-chunks.json")
# [{"chunk_id": "stpo-001", "text": "...", "metadata": {...}}, ...]
```

#### `validate_corpus(corpus)`

Validate that every dict in the corpus has the required keys (`chunk_id`, `text`, `metadata`). Raises `ValueError` with a descriptive message if any dict is missing required keys. Called automatically by `BM25Retriever.index()` and `VectorRetriever.index()`.

### Reciprocal Rank Fusion (RRF)

#### `rrf_fuse(result_lists, k_rrf, k, weights)`

Fuse multiple ranked result lists into a single ranking using Reciprocal Rank Fusion (Cormack et al., 2009). Used by both `HybridRetriever` (within-KB) and `FusionRetriever` (cross-KB).

```
RRF_score(d) = Σ  weight_i / (k_rrf + rank_i(d))
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result_lists` | `list[list[RetrievalResult]]` | — | Ranked result lists to fuse. |
| `k_rrf` | `int` | `60` | RRF smoothing constant. Higher values flatten the rank distribution. |
| `k` | `int` | `5` | Number of top results to return. |
| `weights` | `list[float] \| None` | `None` | Optional per-list weights (uniform if None). |

---

## Generation

### Generator

Abstract base class for all generation strategies. The concrete `OllamaGenerator` is defined in `src/marley/generator/`.

```python
class Generator(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def generate(self, query: str, context: list[dict[str, Any]]) -> GenerationResult: ...
```

| Member | Description |
|---|---|
| `model` | Abstract property returning the model identifier (e.g., `"llama3.1:latest"`). |
| `generate(query, context)` | Generate an answer. `query` is the user question; `context` is a list of chunk dicts with `chunk_id`, `text`, and optionally `metadata`. |

### GenerationResult

Output of a single generation call.

```python
@dataclass
class GenerationResult:
    answer: str                                          # Generated answer text
    model: str                                           # LLM model identifier
    context_chunk_ids: list[str] = field(default_factory=list)  # Chunk IDs used as context
    prompt_tokens: int = 0                               # Tokens in the prompt
    completion_tokens: int = 0                           # Tokens generated
```

---

## Abstention

### AbstentionResult

Result of the abstention-aware pipeline. Captures whether the system abstained, at which level, and why.

- **Level 1** abstention: retrieval confidence below threshold.
- **Level 2** abstention: LLM explicitly signals insufficient context.

```python
@dataclass
class AbstentionResult:
    abstained: bool                                                  # Whether the system abstained
    level: int | None                                                # 1 = retrieval, 2 = LLM, None = answered
    reason: str                                                      # Abstention reason ("" if answered)
    answer: str                                                      # Generated answer ("" if abstained)
    confidence: float                                                # Top-1 normalized retrieval score
    retrieval_results: list[dict[str, Any]] = field(default_factory=list)  # Raw retrieval results
    model: str = ""                                                  # LLM model identifier
```

---

## Quality

### QualityFlag

A diagnostic flag raised during pipeline processing. Used by the extractor, both chunkers, and potentially later stages to record non-fatal issues.

```python
@dataclass
class QualityFlag:
    code: str                        # Machine-readable code (e.g., "EMPTY_SECTION")
    message: str                     # Human-readable description
    severity: str                    # "info", "warning", or "error"
    context: dict[str, Any] = field(default_factory=dict)  # Additional context
```

---

## Scoring

Three utility functions for score normalization, filtering, and confidence computation. Used by the abstention module to apply a uniform confidence threshold across different retriever types.

### `normalize_scores(results, strategy, *, bm25_k, rrf_n_retrievers, rrf_k)`

Normalize retrieval scores to the [0, 1] range.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `list[RetrievalResult]` | — | Raw retrieval results. |
| `strategy` | `str` | — | `"bm25"`, `"vector"`, or `"rrf"`. |
| `bm25_k` | `float` | `1.0` | Saturation parameter for BM25 normalization. |
| `rrf_n_retrievers` | `int` | `2` | Number of sub-retrievers (for RRF max-score). |
| `rrf_k` | `int` | `60` | RRF smoothing constant. |

**Normalization strategies:**

| Strategy | Formula | Rationale |
|---|---|---|
| `bm25` | `score / (score + k)` | Saturation function mapping unbounded BM25 scores to [0, 1]. |
| `vector` | identity | Cosine similarity already in [0, 1]. |
| `rrf` | `score / max_score` | Divide by theoretical RRF maximum `n / (k_rrf + 1)`. |

### `filter_by_threshold(results, threshold)`

Remove results with score below the given threshold. Expects normalized scores.

| Parameter | Type | Description |
|---|---|---|
| `results` | `list[RetrievalResult]` | Normalized retrieval results. |
| `threshold` | `float` | Minimum score to retain a result. |

### `compute_confidence(results)`

Return the maximum score from the result set. Represents the system's confidence that at least one retrieved chunk is relevant. Returns `0.0` for an empty result set.

---

## Utility Functions

### `compute_token_stats(token_counts)`

Compute min, median, max, and total token statistics from a list of token counts. Used by both the PDF chunker and the FAQ chunker.

```python
from src.marley.models import compute_token_stats

stats = compute_token_stats([100, 200, 300])
# {"min_tokens": 100, "median_tokens": 200, "max_tokens": 300, "total_tokens": 600}
```

Returns all zeros for an empty list.

### `save_json(result, output_path)`

Serialize a dataclass instance to a JSON file via `dataclasses.asdict`. Creates parent directories if they do not exist.

```python
from src.marley.models import save_json

path = save_json(extraction_result, "data/knowledgebase/stpo-extracted.json")
```

Returns the resolved absolute path of the written file.

---

## Constants

All constants are defined in `src/marley/models/constants.py` and imported by the relevant modules. They are not re-exported from `__init__.py` but constitute the pipeline's default configuration.

### Retrieval

| Constant | Type | Value | Description |
|---|---|---|---|
| `DEFAULT_K` | `int` | `5` | Default number of chunks to retrieve. |
| `DEFAULT_K_RRF` | `int` | `60` | General RRF smoothing constant (Cormack et al., 2009). |
| `DEFAULT_K_RRF_HYBRID` | `int` | `60` | RRF smoothing constant for HybridRetriever (BM25 + Vector). |
| `DEFAULT_K_RRF_FUSION` | `int` | `60` | RRF smoothing constant for FusionRetriever (cross-KB). |
| `RETRIEVER_TYPES` | `list[str]` | `["bm25", "vector", "hybrid"]` | Supported retriever type identifiers. |
| `STRATEGIES` | `list[str]` | `["single", "merged_pool", "fusion"]` | Supported knowledge-base combination strategies. |

### Score Normalization

| Constant | Type | Value | Description |
|---|---|---|---|
| `NORMALIZATION_STRATEGIES` | `set[str]` | `{"bm25", "vector", "rrf"}` | Recognized normalization strategy names. |
| `NORMALIZATION_MAP` | `dict[str, str]` | `{"bm25": "bm25", "vector": "vector", "hybrid": "rrf"}` | Mapping from retriever type to normalization strategy. |
| `DEFAULT_THRESHOLD` | `float` | `0.3` | Default abstention confidence threshold. |
| `DEFAULT_THRESHOLDS` | `dict[str, float]` | `{"bm25": 0.3, "vector": 0.3, "rrf": 0.3}` | Per-strategy default thresholds. |

### Chunking

| Constant | Type | Value | Description |
|---|---|---|---|
| `DEFAULT_MAX_CHUNK_TOKENS` | `int` | `512` | Maximum token count per chunk. |
| `DEFAULT_MIN_CHUNK_TOKENS` | `int` | `64` | Minimum token count per chunk (smaller chunks are merged). |
| `DEFAULT_OVERLAP_TOKENS` | `int` | `50` | Token overlap between consecutive sliding-window chunks. |
| `DEFAULT_TOKENIZER` | `str` | `"cl100k_base"` | Default tiktoken encoding for token counting. |

### Vector Retrieval

| Constant | Type | Value | Description |
|---|---|---|---|
| `CHROMADB_BATCH_SIZE` | `int` | `5000` | Maximum batch size for ChromaDB add operations. |

### Server

| Constant | Type | Value | Description |
|---|---|---|---|
| `SOURCE_TEXT_TRUNCATION` | `int` | `500` | Maximum character length for source text snippets in API responses. |

---

## Usage

```python
from src.marley.models import (
    # Extraction
    ExtractionResult, Section, Table,
    # Quality
    QualityFlag,
    # Retrieval
    Retriever, RetrievalResult, load_chunks, validate_corpus, rrf_fuse,
    # Generation
    Generator, GenerationResult,
    # Abstention
    AbstentionResult,
    # Scoring
    normalize_scores, filter_by_threshold, compute_confidence,
    # Utilities
    compute_token_stats, save_json,
)
```

All symbols are re-exported from `src.marley.models`, so downstream modules never need to import from submodules directly. Constants are accessed via direct import from `src.marley.models.constants`.
