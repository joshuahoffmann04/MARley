# Retrieval Overview

**Package:** `src/marley/retrieval/`
**Models:** `src/marley/models/retrieval.py`, `src/marley/models/constants.py`

The retrieval module implements a strategy pattern with an abstract `Retriever` base class. All retriever implementations share the same interface, enabling the pipeline to swap between sparse, dense, and hybrid strategies without changing downstream code.

---

## Class Hierarchy

All retriever implementations inherit from the `Retriever` abstract base class defined in `src/marley/models/retrieval.py`.

```
Retriever (abstract)
├── index(corpus) → None
├── retrieve(query, k) → list[RetrievalResult]
└── size → int

BM25Retriever(Retriever)        → bm25.md
└── Sparse retrieval via rank_bm25.BM25Okapi

VectorRetriever(Retriever)      → vector.md
└── Dense retrieval via sentence-transformers + ChromaDB

HybridRetriever(Retriever)      → hybrid.md
├── Wraps exactly two Retriever instances (within-KB)
└── Fuses ranked lists via rrf_fuse()

MergedRetriever(Retriever)      → merged.md
├── Wraps a single inner Retriever instance (multi-KB)
└── Delegates all operations to the inner retriever

FusionRetriever(Retriever)      → fusion.md
├── Wraps N pre-indexed Retriever instances (cross-KB)
└── Fuses ranked lists via rrf_fuse()
```

### Design Rationale

- **Strategy pattern:** A common `Retriever` interface allows the evaluation framework, server, and end-to-end pipeline to treat all retrieval strategies uniformly.
- **Composition over inheritance:** HybridRetriever and FusionRetriever wrap other `Retriever` instances via dependency injection rather than extending them. This decouples strategy selection from corpus management.
- **Separation of concerns:** BM25 and Vector are leaf retrievers (they own an index). Hybrid, Merged, and Fusion are composite retrievers (they delegate to leaf retrievers).

---

## Retriever Interface

All retrievers implement three operations:

| Method | Description |
|---|---|
| `index(corpus)` | Build the retrieval index from a list of chunk dicts. Each dict must contain `chunk_id`, `text`, and `metadata`. |
| `retrieve(query, k)` | Return top-k results sorted by descending relevance score. |
| `size` | Number of indexed chunks (read-only property). |

**Exceptions:** `FusionRetriever.index()` raises `NotImplementedError` because its sub-retrievers are pre-indexed against separate corpora. `MergedRetriever.index()` delegates to its inner retriever.

### RetrievalResult

Results are returned as `RetrievalResult` dataclasses:

```python
@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
```

### Score Scales

Different retriever types produce scores on different scales:

| Retriever | Score Range | Higher = Better |
|---|---|---|
| BM25 | 0 to ~100+ (unbounded) | Yes |
| Vector | -1.0 to 1.0 (cosine similarity) | Yes |
| Hybrid / Fusion | 0 to ~0.03 (RRF score) | Yes |

This incompatibility is intentional — RRF-based fusion operates on ranks, not scores, avoiding the calibration problem entirely.

### Score Normalization

Because raw scores live on different scales, a normalization step maps them to [0, 1] before they can be used as confidence values for abstention decisions. Implementation: `src/marley/models/scoring.py`.

| Retriever | Strategy | Formula | Behavior |
|---|---|---|---|
| BM25 | `bm25` | `s / (s + k)` (k=1.0) | Saturation: asymptotic, never reaches 1.0 |
| Vector | `vector` | identity | Cosine similarity already in [0, 1] |
| Hybrid/Fusion | `rrf` | `s / max_rrf` | Divides by theoretical maximum `n_retrievers / (k_rrf + 1)` |

#### Normalization examples for the same highly-relevant chunk

| Strategy | Raw Score | Normalized | Why |
|---|---|---|---|
| BM25 | 2.3 | 0.70 | `2.3 / (2.3 + 1.0)` — saturation compresses |
| Vector | 0.98 | 0.98 | Identity — already calibrated |
| RRF (k=60, 2 retrievers) | 0.0328 | 1.00 | Rank #1 in both → equals theoretical max `2/61` |

#### Practical consequence

Normalized confidences are **not comparable across retriever types**. A confidence of 0.7 from BM25 represents a very different level of certainty than 0.7 from RRF. In particular:

- **RRF scores cluster near the maximum** because raw RRF values span a very narrow range (~0.015–0.033 with k=60). Any chunk ranked #1 in at least one sub-retriever will normalize close to 1.0.
- **BM25 confidences are structurally lower** because the saturation function compresses the unbounded score range.
- **Vector confidences vary naturally** with cosine similarity, producing the most discriminative spread.

This is why the abstention evaluation runs a **per-configuration threshold sweep** — the optimal threshold differs per retriever type.

---

## Corpus Utilities

Shared functions in `src/marley/models/retrieval.py`, re-exported from `src.marley.retrieval`:

| Function | Description |
|---|---|
| `load_chunks(chunk_path)` | Load chunks from a JSON file. Returns a list of dicts with `chunk_id`, `text`, `metadata`. |
| `validate_corpus(corpus)` | Validate that every dict has the required keys. Raises `ValueError` on missing keys. Called by `BM25Retriever.index()` and `VectorRetriever.index()`. |

---

## Reciprocal Rank Fusion (RRF)

Both HybridRetriever and FusionRetriever use the same `rrf_fuse()` function (defined in `src/marley/models/retrieval.py`):

```
RRF_score(d) = Σ  weight_i / (k_rrf + rank_i(d))
```

- **Rank-based:** Uses rank positions, not raw scores. Avoids score normalization between retrievers with incompatible score distributions.
- **Monotonic:** A document that improves its rank in any input list cannot decrease its fused score.
- **Weighted:** Optional per-retriever/per-KB weights (default: uniform).
- **k_rrf:** Smoothing constant (default: 60, from Cormack et al., 2009). A sweep over [1-100] showed no measurable impact on Recall@5 (see [rrf-tuning.md](../evaluation/rrf-tuning.md)).

| Use Case | Class | Sub-retrievers | Purpose |
|---|---|---|---|
| Within-KB fusion | HybridRetriever | Exactly 2 (e.g., BM25 + Vector) | Combine sparse + dense on the same corpus |
| Multi-KB merged pool | MergedRetriever | 1 (inner retriever) | Concatenate KBs into one corpus, single ranking |
| Multi-KB cross-KB fusion | FusionRetriever | N >= 1 (one per KB) | Per-KB ranking, fused via RRF |

---

## Configuration Constants

Defined in `src/marley/models/constants.py`:

| Constant | Value | Description |
|---|---|---|
| `DEFAULT_K` | 5 | Top-k results to return |
| `DEFAULT_K_RRF_HYBRID` | 60 | k_rrf for HybridRetriever |
| `DEFAULT_K_RRF_FUSION` | 60 | k_rrf for FusionRetriever |
| `CHROMADB_BATCH_SIZE` | 5000 | Max batch size for ChromaDB inserts |
| `RETRIEVER_TYPES` | bm25, vector, hybrid | Supported retriever identifiers |
| `STRATEGIES` | single, merged_pool, fusion | KB combination strategies |

---

## Module Structure

```
src/marley/
├── models/
│   ├── retrieval.py       # Retriever ABC, RetrievalResult, load_chunks, validate_corpus
│   └── constants.py       # DEFAULT_K, DEFAULT_K_RRF_*, CHROMADB_BATCH_SIZE
└── retrieval/
    ├── __init__.py        # Public API (re-exports all symbols)
    ├── base.py            # Re-export of Retriever, RetrievalResult from models
    ├── bm25.py            # BM25Retriever
    ├── vector.py          # VectorRetriever
    ├── hybrid.py          # HybridRetriever (within-KB RRF)
    ├── merged.py          # MergedRetriever (multi-KB merged pool)
    └── fusion.py          # FusionRetriever (cross-KB RRF), re-exports rrf_fuse()
```

---

## Imports

All public symbols are available from the package root:

```python
from src.marley.retrieval import (
    BM25Retriever,
    VectorRetriever,
    HybridRetriever,
    FusionRetriever,
    MergedRetriever,
    Retriever,
    RetrievalResult,
    rrf_fuse,
    load_chunks,
    validate_corpus,
)
```

---

## Documentation

| Document | Covers |
|---|---|
| [bm25.md](bm25.md) | BM25 sparse retrieval: theory, tokenization, scoring |
| [vector.md](vector.md) | Dense vector retrieval: embeddings, ChromaDB, persistence |
| [hybrid.md](hybrid.md) | Within-KB RRF fusion: BM25 + Vector combination |
| [merged.md](merged.md) | Merged-pool strategy: multi-KB single corpus |
| [fusion.md](fusion.md) | Cross-KB RRF fusion: per-KB ranking fused via RRF |

---

## Glossary

| Term | Meaning |
|---|---|
| **Chunk** | Atomic text unit produced by the chunking pipeline. Each chunk has a `chunk_id`, `text`, and `metadata`. Chunks are the items stored in retrieval indices. |
| **Corpus** | A list of chunk dicts passed to `index()`. May contain chunks from one or multiple knowledge bases. |
| **Knowledge base (KB)** | A named source of chunks (e.g., `stpo`, `faq-stpo`, `faq-ao`). Each KB has its own chunk file. |
| **Sub-retriever** | A `Retriever` instance wrapped by HybridRetriever or FusionRetriever. |
| **Score** | Raw relevance value from a retriever. Scale depends on retriever type (see Score Scales above). |
| **Confidence** | Normalized score (0–1) used for abstention decisions. Computed from raw scores via `normalize_scores()`. |
