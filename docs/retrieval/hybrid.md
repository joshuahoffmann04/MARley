# Hybrid Retrieval (Reciprocal Rank Fusion)

**Module:** `src/marley/retrieval/`
**Implementation:** `hybrid.py`
**Dependencies:** None beyond the base retrieval module (uses injected `Retriever` instances)

Hybrid retrieval combines the ranked results of two retriever instances using Reciprocal Rank Fusion (RRF). This merges the complementary strengths of sparse (BM25) and dense (Vector) retrieval into a single ranked list.

---

## Architecture

```
Retriever (abstract)
├── index(corpus) → None
├── retrieve(query, k) → list[RetrievalResult]
└── size → int

BM25Retriever(Retriever)
└── Uses rank_bm25.BM25Okapi internally

VectorRetriever(Retriever)
└── Uses sentence-transformers + ChromaDB internally

HybridRetriever(Retriever)
├── Wraps exactly two Retriever instances (dependency injection)
└── Fuses ranked lists using Reciprocal Rank Fusion (RRF)
```

---

## Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists into a single ranking by assigning each document a fused score based on its rank in each list:

```
RRF_score(d) = Σ  1 / (k_rrf + rank_i(d))
```

where:
- `rank_i(d)` is the 1-based rank of document `d` in retriever `i`'s result list
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

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `retrievers` | *(required)* | Tuple of exactly two `Retriever` instances. |
| `k_rrf` | `60` | RRF smoothing constant. Higher values flatten the rank distribution. |

---

## Result Handling

- Each sub-retriever is queried for `k` results.
- Documents appearing in both result lists receive combined RRF scores.
- The final output is sorted by RRF score descending, limited to `k` results.
- When a document appears in both lists, the text and metadata from the **higher-scoring** source are used.
- The `score` field in results contains the RRF fused score (not the original retriever scores).
- The `size` property returns the size of the first sub-retriever (both should contain the same corpus).

---

## Imports

```python
from src.marley.retrieval import HybridRetriever, BM25Retriever, VectorRetriever, Retriever, RetrievalResult, load_chunks
```
