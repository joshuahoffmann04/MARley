# Vector Retrieval

**Module:** `src/marley/retrieval/`
**Implementation:** `vector.py`
**Libraries:** `sentence-transformers` (all-mpnet-base-v2), `chromadb` (persistent vector store)

Dense vector retrieval using sentence embeddings and cosine similarity. Complements the sparse BM25 baseline by capturing semantic similarity rather than keyword overlap.

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
└── Fuses two Retriever instances via RRF (see docs/retrieval/hybrid.md)
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

Each knowledge base uses its own `persist_directory`:

```
data/vectorstore/
├── stpo/           # StPO chunks
├── faq-stpo/       # FAQ-StPO chunks
└── faq-ao/         # FAQ-AO chunks
```

- `index()` computes embeddings and stores them persistently via ChromaDB.
- On subsequent instantiation with the same `persist_directory`, the existing index is loaded automatically — no re-embedding required.
- `index()` on an existing store replaces the collection entirely (clean re-index).
- `data/vectorstore/` is gitignored (derived data).

---

## Score Conversion

ChromaDB returns cosine distances (0 = identical, 2 = opposite). These are converted to similarity scores:

```
score = 1.0 - distance
```

This produces scores in the range [-1, 1], where 1 means identical and -1 means opposite. Unlike BM25, all results are returned (no zero-score filtering), since even low-similarity results may carry useful semantic information.

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
from src.marley.retrieval import VectorRetriever, Retriever, RetrievalResult, load_chunks
```
