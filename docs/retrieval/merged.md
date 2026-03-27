# Merged-Pool Retrieval (Multi-KB Single Corpus)

**Module:** `src/marley/retrieval/`
**Implementation:** `merged.py`
**Dependencies:** None beyond the base retrieval module (uses an injected `Retriever` instance)

Merged-pool retrieval is the simplest multi-KB strategy: chunks from all selected knowledge bases are concatenated into a single corpus and indexed by one inner retriever. The inner retriever (BM25, Vector, or Hybrid) searches across all KBs simultaneously without any per-KB distinction.

---

## Architecture

For the full retriever class hierarchy and shared interface, see [overview.md](overview.md).

```
MergedRetriever(Retriever)
+-- Wraps a single inner Retriever instance
+-- index(corpus) -> delegates to inner retriever
+-- retrieve(query, k) -> delegates to inner retriever
+-- size -> delegates to inner retriever
```

---

## How It Works

1. Chunks from all selected knowledge bases are loaded and concatenated into one list.
2. The merged list is passed to `index()`, which delegates to the inner retriever.
3. `retrieve()` queries the inner retriever, which searches the entire merged corpus.

The merge happens at the data level -- the inner retriever is unaware that chunks come from different KBs. This means ranking is based purely on relevance to the query, without per-KB boundaries.

---

## Key Differences from FusionRetriever

| Aspect | MergedRetriever | FusionRetriever |
|---|---|---|
| Purpose | Search all KBs as one corpus | Fuse per-KB rankings via RRF |
| Sub-retrievers | 1 (wraps a single inner retriever) | N >= 1 (one per KB) |
| `index()` | Delegates to inner retriever | Raises `NotImplementedError` |
| `size` | Inner retriever's size | Sum of all sub-retriever sizes |
| Ranking | Single global ranking | RRF fusion of per-KB rankings |
| Score type | Inner retriever's native scores | RRF scores |
| KB awareness | None -- all chunks treated equally | Per-KB results before fusion |

### When to Use Which

- **MergedRetriever** when KB boundaries don't matter and you want the simplest approach. Works well when KBs have similar content density.
- **FusionRetriever** when you want each KB to contribute independently to the final ranking, avoiding large KBs dominating results.

---

## Usage

```python
from src.marley.retrieval import MergedRetriever, BM25Retriever, load_chunks

# Load chunks from multiple KBs
stpo_chunks = load_chunks("data/chunks/stpo-chunks.json")
faq_chunks = load_chunks("data/chunks/faq-stpo-chunks.json")

# Merge and index
inner = BM25Retriever()
merged = MergedRetriever(inner)
merged.index(stpo_chunks + faq_chunks)

# Retrieve across all KBs
results = merged.retrieve("master thesis credits", k=5)
```

### With Hybrid Inner Retriever

```python
from src.marley.retrieval import MergedRetriever, HybridRetriever, BM25Retriever, VectorRetriever

inner = HybridRetriever(retrievers=(
    BM25Retriever(),
    VectorRetriever(persist_directory="data/vectorstore/merged"),
))
merged = MergedRetriever(inner)
merged.index(stpo_chunks + faq_chunks)
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `retriever` | *(required)* | Inner `Retriever` instance (BM25, Vector, or Hybrid). |

MergedRetriever itself has no additional parameters. All configuration (e.g., `k_rrf` for a Hybrid inner retriever) is set on the inner retriever.

---

## Design Decisions

1. **Thin wrapper by design.** MergedRetriever adds no algorithm of its own -- it is an architectural marker that makes the merged-pool strategy explicit in the type system and symmetric with FusionRetriever.

2. **Separation from service layer.** Previously, merged-pool logic was implicit in `PipelineService.get_retriever()`. Extracting it into a dedicated class keeps the server layer free of retrieval strategy logic.

3. **Standard `Retriever` interface.** MergedRetriever implements the full `Retriever` ABC, so it can be used anywhere a `Retriever` is expected (evaluation, testing, pipeline).

---

## Imports

```python
from src.marley.retrieval import MergedRetriever, Retriever, RetrievalResult, load_chunks, validate_corpus
```
