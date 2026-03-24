# Fusion Retrieval (Cross-KB Reciprocal Rank Fusion)

**Module:** `src/marley/retrieval/`
**Implementation:** `fusion.py`
**Dependencies:** None beyond the base retrieval module (uses injected `Retriever` instances)

Fusion retrieval merges ranked result lists from multiple independently indexed retrievers using Reciprocal Rank Fusion (RRF). Unlike `HybridRetriever`, which fuses two retrieval *strategies* over the same corpus (BM25 + Vector), `FusionRetriever` fuses results across different *knowledge bases* (e.g., StPO + FAQ-StPO + FAQ-AO), each served by its own pre-indexed retriever.

The module also exposes the standalone `rrf_fuse()` function, which is shared by both `HybridRetriever` and `FusionRetriever`.

---

## Architecture

For the full retriever class hierarchy and shared interface, see [overview.md](overview.md).

```
FusionRetriever(Retriever)
├── Wraps N pre-indexed Retriever instances (N >= 1)
├── Delegates retrieve() to all sub-retrievers
├── Fuses ranked lists via rrf_fuse()
└── index() raises NotImplementedError (sub-retrievers must be pre-indexed)

rrf_fuse()  [standalone function]
├── Used by HybridRetriever (within-KB fusion)
└── Used by FusionRetriever (cross-KB fusion)
```

---

## Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists into a single ranking by assigning each document a fused score based on its rank position in each input list:

```
RRF_score(d) = Σ  weight_i / (k_rrf + rank_i(d))
```

where:
- `rank_i(d)` is the 1-based rank of document `d` in list `i`
- `weight_i` is the per-list weight (default: 1.0 for all lists, i.e., uniform weighting)
- `k_rrf` is a smoothing constant (default: 60)

RRF was proposed by Cormack, Clarke, and Buettcher (2009) and has become a standard method for unsupervised rank fusion. Its key advantage is that it operates on rank positions rather than raw scores, avoiding the score calibration problem that arises when combining retrievers with incompatible score distributions.

### Properties

- **Rank-based:** Uses only rank positions, not raw scores. This eliminates the need to normalize scores across different retrievers or knowledge bases.
- **Monotonic:** A document that improves its rank in any input list cannot decrease its fused score.
- **Bounded:** For `N` uniformly weighted input lists, the maximum possible RRF score is `N / (k_rrf + 1)`. With custom weights, the bound becomes `sum(weights) / (k_rrf + 1)`.
- **k_rrf smoothing:** Higher values reduce the score difference between top-ranked and lower-ranked documents. The default of 60 is the value recommended in the original paper.

---

## `rrf_fuse()` Function

Standalone RRF fusion utility. Accepts any number of ranked result lists and returns a single fused ranking.

```python
def rrf_fuse(
    result_lists: list[list[RetrievalResult]],
    k_rrf: int = DEFAULT_K_RRF,
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result_lists` | `list[list[RetrievalResult]]` | — | Ranked result lists to fuse (any number). |
| `k_rrf` | `int` | `60` | RRF smoothing constant. Higher values flatten the rank distribution. |
| `k` | `int` | `5` | Number of top results to return. |
| `weights` | `list[float] \| None` | `None` (uniform) | Optional per-list weights. Must be positive and match the length of `result_lists`. |

**Behavior:**
- Each input list is assumed to be sorted by descending relevance from its source retriever.
- When a document appears in multiple lists, its text and metadata are taken from the list where it had the highest original score.
- Returns an empty list if `result_lists` is empty.
- The `score` field in the returned results contains the RRF fused score.

---

## `FusionRetriever` Class

Wrapper that manages multiple pre-indexed retrievers and fuses their results via RRF. Designed for combined-KB configurations where each knowledge base has its own dedicated retriever.

### Key Differences from HybridRetriever

| Aspect | HybridRetriever | FusionRetriever |
|---|---|---|
| Purpose | Fuse retrieval *strategies* (BM25 + Vector) | Fuse retrieval across *knowledge bases* |
| Sub-retrievers | Exactly 2 (tuple) | Any number >= 1 (list) |
| `index()` | Delegates to both sub-retrievers | Raises `NotImplementedError` |
| `size` | Size of first sub-retriever | Sum of all sub-retriever sizes |
| Default `k_rrf` | `DEFAULT_K_RRF_HYBRID` (60) | `DEFAULT_K_RRF_FUSION` (60) |
| Corpus | Same corpus in both sub-retrievers | Different corpus per sub-retriever |

### Why `index()` is Not Supported

FusionRetriever wraps retrievers that are already indexed against their respective knowledge bases. Each sub-retriever may have different initialization requirements (e.g., separate `persist_directory` for each VectorRetriever). Calling `index()` on the FusionRetriever would be ambiguous — there is no single corpus to distribute.

---

## Usage

```python
from src.marley.retrieval import HybridRetriever, FusionRetriever, BM25Retriever, VectorRetriever, load_chunks

# Build per-KB hybrid retrievers
stpo_chunks = load_chunks("data/chunks/stpo-chunks.json")
bm25_stpo = BM25Retriever()
bm25_stpo.index(stpo_chunks)
vec_stpo = VectorRetriever(persist_directory="data/vectorstore/stpo")
hybrid_stpo = HybridRetriever(retrievers=(bm25_stpo, vec_stpo))

faq_chunks = load_chunks("data/chunks/faq-stpo-chunks.json")
bm25_faq = BM25Retriever()
bm25_faq.index(faq_chunks)
vec_faq = VectorRetriever(persist_directory="data/vectorstore/faq-stpo")
hybrid_faq = HybridRetriever(retrievers=(bm25_faq, vec_faq))

# Fuse across knowledge bases
fusion = FusionRetriever(retrievers=[hybrid_stpo, hybrid_faq])
results = fusion.retrieve("master thesis credits", k=5)
```

### Weighted Cross-KB Fusion

```python
# Boost the StPO knowledge base (weight 2.0) relative to FAQ (weight 1.0)
fusion = FusionRetriever(
    retrievers=[hybrid_stpo, hybrid_faq],
    weights=[2.0, 1.0],
)
```

### Using `rrf_fuse()` Directly

```python
from src.marley.retrieval import rrf_fuse

fused = rrf_fuse([stpo_results, faq_results], k_rrf=60, k=10)

# With weights:
fused = rrf_fuse([stpo_results, faq_results], weights=[2.0, 1.0], k=10)
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `retrievers` | *(required)* | List of pre-indexed `Retriever` instances (>= 1). |
| `k_rrf` | `60` | RRF smoothing constant. Higher values flatten the rank distribution. |
| `weights` | `None` (uniform) | Optional list of positive floats (one per retriever). Boosts or dampens individual knowledge bases. |

---

## Design Decisions

1. **Separate module from HybridRetriever.** Although both use RRF, they solve different problems: Hybrid fuses strategies within one KB; Fusion fuses results across KBs. Separate classes make the intent explicit and allow independent configuration (e.g., different `k_rrf` defaults).

2. **Shared `rrf_fuse()` function.** The core RRF algorithm is factored into a standalone function to avoid code duplication between `HybridRetriever` and `FusionRetriever`.

3. **Pre-indexed design.** FusionRetriever does not manage indexing because each sub-retriever operates on a different corpus with potentially different configuration. This follows the composition pattern where the caller is responsible for setup.

4. **Variable number of sub-retrievers.** Unlike HybridRetriever (exactly 2), FusionRetriever accepts any number of sub-retrievers. This supports configurations with 2 or 3 knowledge bases without code changes.

---

## Imports

```python
from src.marley.retrieval import FusionRetriever, rrf_fuse, Retriever, RetrievalResult, load_chunks, validate_corpus
```
