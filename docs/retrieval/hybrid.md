# Hybrid Retrieval (Reciprocal Rank Fusion)

**Module:** `src/marley/retrieval/`
**Implementation:** `hybrid.py`
**Dependencies:** None beyond the base retrieval module (uses injected `Retriever` instances)

Hybrid retrieval combines the ranked results of two retriever instances using Reciprocal Rank Fusion (RRF). This merges the complementary strengths of sparse (BM25) and dense (Vector) retrieval into a single ranked list.

---

## Theoretical Background

Sparse and dense retrieval methods have complementary strengths. BM25 excels at exact keyword matching and handles domain-specific terminology well, while dense retrieval captures semantic similarity and can match paraphrases. Empirical studies consistently show that combining both methods outperforms either in isolation (Lin et al., 2021; Ma et al., 2021).

Two families of combination strategies exist:

1. **Score fusion** normalizes and combines raw scores from multiple retrievers. This requires careful calibration because different retrievers produce scores on incompatible scales (BM25 scores are unbounded; cosine similarity lies in [-1, 1]).
2. **Rank fusion** combines ranked lists based on rank positions rather than scores. This avoids the calibration problem entirely.

MARley uses **Reciprocal Rank Fusion (RRF)**, a rank fusion method proposed by Cormack, Clarke, and Buettcher (2009). RRF assigns each document a fused score based on its rank in each input list: `score(d) = sum(1 / (k + rank_i(d)))`, where `k` is a smoothing constant (default 60). The method is simple, parameter-light, and has been shown to outperform both individual rankers and more complex fusion methods such as Condorcet voting (Cormack et al., 2009).

The HybridRetriever applies RRF to fuse BM25 and Vector results over the **same corpus** (within-KB fusion). For cross-KB fusion, see [fusion.md](fusion.md).

---

## Architecture

For the full retriever class hierarchy and shared interface, see [overview.md](overview.md).

```
HybridRetriever(Retriever)
├── Wraps exactly two Retriever instances (dependency injection)
├── index(corpus) → delegates to both sub-retrievers
├── retrieve(query, k) → queries both, fuses via rrf_fuse()
└── size → first sub-retriever's size
```

---

## Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists into a single ranking by assigning each document a fused score based on its rank in each list:

```
RRF_score(d) = Σ  weight_i / (k_rrf + rank_i(d))
```

where:
- `rank_i(d)` is the 1-based rank of document `d` in retriever `i`'s result list
- `weight_i` is the per-retriever weight (default: 1.0 for all, i.e., uniform weighting)
- `k_rrf` is a smoothing constant (default: 60, from the original RRF paper by Cormack et al., 2009)

### Properties

- **Rank-based:** RRF uses only rank positions, not raw scores. This avoids score normalization issues between retrievers with different score scales (e.g., BM25 unbounded scores vs. cosine similarity in [-1, 1]).
- **Complementary fusion:** Documents found by both retrievers receive scores from both lists, naturally ranking higher than documents found by only one.
- **k_rrf smoothing:** Higher values reduce the influence of top-ranked documents relative to lower-ranked ones. The default of 60 is well-established in the literature.

---

## Design: Dependency Injection

The HybridRetriever receives two pre-configured `Retriever` instances in its constructor. It does not create retrievers internally.

**Rationale:**
- Each sub-retriever may require different initialization (e.g., VectorRetriever needs a `persist_directory`)
- Sub-retrievers can be reused independently outside the hybrid context
- Easier to test with mock/fake retrievers
- Follows the composition-over-inheritance principle

---

## Usage

```python
from src.marley.retrieval import BM25Retriever, VectorRetriever, HybridRetriever, load_chunks

# Setup sub-retrievers
chunks = load_chunks("data/chunks/stpo-chunks.json")

bm25 = BM25Retriever()
bm25.index(chunks)

vector = VectorRetriever(persist_directory="data/vectorstore/stpo")
# (vector index already exists from previous indexing)

# Create hybrid retriever
hybrid = HybridRetriever(retrievers=(bm25, vector))

# Retrieve
results = hybrid.retrieve("master thesis credits", k=5)
for r in results:
    print(f"{r.chunk_id}: {r.score:.4f}")
```

### Using `index()` to Build Both Sub-Retrievers

```python
hybrid = HybridRetriever(retrievers=(BM25Retriever(), VectorRetriever(persist_directory="data/vectorstore/stpo")))
hybrid.index(chunks)  # Delegates to both sub-retrievers
```

### Weighted Fusion

```python
# Boost the vector retriever (weight 2.0) relative to BM25 (weight 1.0)
hybrid = HybridRetriever(
    retrievers=(bm25, vector),
    weights=[1.0, 2.0],
)
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `retrievers` | *(required)* | Tuple of exactly two `Retriever` instances. |
| `k_rrf` | `60` | RRF smoothing constant. Higher values flatten the rank distribution. |
| `weights` | `None` (uniform) | Optional list of two positive floats. Boosts or dampens the contribution of each retriever. |

---

## Result Handling

- Each sub-retriever is queried for `k` results.
- Documents appearing in both result lists receive combined RRF scores.
- The final output is sorted by RRF score descending, limited to `k` results.
- When a document appears in both lists, the text and metadata from the **higher-scoring** source are used.
- The `score` field in results contains the RRF fused score (not the original retriever scores). RRF scores are typically in the range 0 to ~0.03, depending on `k_rrf`.
- The `size` property returns the size of the first sub-retriever (both should contain the same corpus).

---

## Imports

```python
from src.marley.retrieval import HybridRetriever, BM25Retriever, VectorRetriever, Retriever, RetrievalResult, load_chunks
```
