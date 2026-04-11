# Vector Retrieval

**Module:** `src/marley/retrieval/`
**Implementation:** `vector.py`
**Libraries:** `sentence-transformers` (all-mpnet-base-v2), `chromadb` (persistent vector store)

Dense vector retrieval using sentence embeddings and cosine similarity. Complements the sparse BM25 baseline by capturing semantic similarity rather than keyword overlap.

---

## Theoretical Background

Dense retrieval maps queries and documents into a shared vector space where semantic similarity is measured by distance metrics (typically cosine similarity or dot product). Unlike sparse methods such as BM25, dense retrieval captures meaning beyond exact term overlap, enabling it to match synonyms, paraphrases, and semantically related concepts.

The foundation of this approach is **sentence embeddings** — fixed-dimensional vector representations of text passages. Reimers and Gurevych (2019) introduced Sentence-BERT (SBERT), which fine-tunes BERT-based models using siamese and triplet network structures to produce embeddings where semantically similar sentences have high cosine similarity. The `all-mpnet-base-v2` model used in MARley is a 768-dimensional SBERT variant based on the MPNet architecture (Song et al., 2020), trained on over 1 billion sentence pairs. It represents the current quality baseline in the `sentence-transformers` library.

For storage and retrieval, MARley uses **ChromaDB**, an embedded vector database that persists embeddings to disk and supports efficient approximate nearest-neighbor search. ChromaDB uses cosine distance internally, which is converted to similarity scores (`1 - distance`) for consistency with the pipeline's scoring conventions.

A key limitation of dense retrieval is its dependence on the embedding model's training data distribution. Domain-specific terminology (e.g., German legal terms, credit point abbreviations) may not be well-represented in general-purpose models. This is why MARley combines dense retrieval with BM25 in the hybrid configuration — the sparse retriever catches exact-match terms that the dense model may miss.

---

## Architecture

For the full retriever class hierarchy and shared interface, see [overview.md](overview.md).

```
VectorRetriever(Retriever)
├── Loads sentence-transformers model lazily on first use
├── Stores/loads embeddings via ChromaDB (persistent)
├── index(corpus) → embeds + stores all chunks
├── retrieve(query, k) → cosine similarity search
└── size → ChromaDB collection count
```

---

## Embedding Model

| Property | Value |
|---|---|
| Model | `sentence-transformers/all-mpnet-base-v2` |
| Dimensions | 768 |
| Max sequence length | 384 tokens |
| Similarity metric | Cosine similarity |
| Size | ~420 MB |

The model is loaded lazily on first use (either `index()` or `retrieve()`).

---

## Persistence

Each knowledge base uses its own `persist_directory`. When used via the server's `PipelineService`, persist directories are created under `data/chunks/.chromadb/` (e.g., `data/chunks/.chromadb/vector-stpo/`). For standalone usage, any path can be specified.

- `index()` validates the corpus via `validate_corpus()`, computes embeddings, and stores them persistently via ChromaDB. Large corpora are inserted in batches of `CHROMADB_BATCH_SIZE` (5000).
- On subsequent instantiation with the same `persist_directory`, the existing index is loaded automatically — no re-embedding required.
- `index()` on an existing store replaces the collection entirely (clean re-index).
- Persist directories are gitignored (derived data).

---

## Score Conversion

ChromaDB returns cosine distances (0 = identical, 2 = opposite). These are converted to similarity scores:

```
score = 1.0 - distance
```

This produces scores in the theoretical range [-1, 1], where 1 means identical and -1 means maximally dissimilar. In practice, scores for real text queries are typically positive (0.0 to 0.8), since natural language texts are rarely semantically opposite. Unlike BM25, all results are returned (no zero-score filtering), since even low-similarity results may carry useful semantic information. If `k <= 0`, an empty list is returned immediately.

---

## Metadata Handling

ChromaDB only supports flat metadata values (str, int, float, bool). The `_flatten_metadatas` helper converts:

| Original type | Stored as | Example |
|---|---|---|
| `None` | `""` (empty string) | `parent_section_id: null` → `""` |
| `list` | `" > "`-joined string | `["Part I", "§23"]` → `"Part I > §23"` |
| Other | `str(value)` | Fallback for unexpected types |
| Empty dict | `None` | ChromaDB rejects empty dicts |

---

## Usage

```python
from src.marley.retrieval import VectorRetriever, load_chunks

# Load and index (first time: embeds + stores)
chunks = load_chunks("data/chunks/stpo-chunks.json")
retriever = VectorRetriever(persist_directory="data/vectorstore/stpo")
retriever.index(chunks)

# Retrieve
results = retriever.retrieve("master thesis credits", k=5)
for r in results:
    print(f"{r.chunk_id}: {r.score:.3f}")
```

### Loading from Existing Store

```python
# No index() needed if persist_directory already contains embeddings
retriever = VectorRetriever(persist_directory="data/vectorstore/stpo")
results = retriever.retrieve("standard study period", k=5)
```

### Separate Knowledge Bases

```python
stpo_retriever = VectorRetriever(persist_directory="data/vectorstore/stpo")
faq_retriever = VectorRetriever(persist_directory="data/vectorstore/faq-stpo")
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `persist_directory` | *(required)* | Path to the ChromaDB storage directory. |
| `model_name` | `sentence-transformers/all-mpnet-base-v2` | Sentence-transformer model for embedding. |
| `collection_name` | `chunks` | ChromaDB collection name (internal). |
| `k` | 5 | Number of top results to return (per query). |

---

## Imports

```python
from src.marley.retrieval import VectorRetriever, Retriever, RetrievalResult, load_chunks, validate_corpus
```
