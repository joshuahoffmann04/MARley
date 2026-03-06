"""Vector retrieval strategy for the MARley pipeline.

Uses sentence-transformers for embedding and ChromaDB for persistent
vector storage with cosine similarity search.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.marley.retrieval.base import RetrievalResult, Retriever

_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
_DEFAULT_COLLECTION = "chunks"


class VectorRetriever(Retriever):
    """Dense vector retrieval over chunked documents.

    Uses a sentence-transformer model to embed chunks and queries,
    and ChromaDB for persistent vector storage and cosine similarity search.
    Each knowledge base should use its own persist_directory.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        model_name: str = _DEFAULT_MODEL,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._model_name = model_name
        self._collection_name = collection_name

        self._model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

        # Load existing persistent store if available
        if self._persist_directory.exists():
            self._connect()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _connect(self) -> None:
        """Connect to the persistent ChromaDB store."""
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
        )
        existing = [c.name for c in self._client.list_collections()]
        if self._collection_name in existing:
            self._collection = self._client.get_collection(
                name=self._collection_name,
            )
        else:
            self._collection = None

    def index(self, corpus: list[dict]) -> None:
        """Embed and store chunks in a persistent ChromaDB collection.

        Each dict must have at least 'chunk_id', 'text', and 'metadata'.
        Replaces any existing collection in the same persist_directory.
        """
        if not corpus:
            # Clear existing collection if any
            if self._client is not None and self._collection is not None:
                self._client.delete_collection(self._collection_name)
                self._collection = None
            return

        model = self._get_model()

        # Ensure persist directory exists
        self._persist_directory.mkdir(parents=True, exist_ok=True)

        # Connect (or reconnect) to ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
        )

        # Delete existing collection to ensure clean re-index
        existing = [c.name for c in self._client.list_collections()]
        if self._collection_name in existing:
            self._client.delete_collection(self._collection_name)

        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Prepare data
        ids = [doc["chunk_id"] for doc in corpus]
        texts = [doc["text"] for doc in corpus]
        metadatas = _flatten_metadatas([doc.get("metadata", {}) for doc in corpus])

        # Embed all texts at once
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # ChromaDB has a batch size limit; add in batches of 5000
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k chunks most relevant to the query."""
        if self._collection is None or self._collection.count() == 0:
            return []

        model = self._get_model()
        query_embedding = model.encode([query], show_progress_bar=False).tolist()

        raw = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []
        for i in range(len(raw["ids"][0])):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - distance
            distance = raw["distances"][0][i]
            score = 1.0 - distance

            results.append(RetrievalResult(
                chunk_id=raw["ids"][0][i],
                text=raw["documents"][0][i],
                score=score,
                metadata=raw["metadatas"][0][i] or {},
            ))

        return results

    @property
    def size(self) -> int:
        """Return the number of indexed documents."""
        if self._collection is None:
            return 0
        return self._collection.count()


def _flatten_metadatas(metadatas: list[dict]) -> list[dict | None]:
    """Flatten metadata dicts for ChromaDB compatibility.

    ChromaDB only supports str, int, float, and bool values.
    Lists and None values must be converted.
    """
    flat = []
    for meta in metadatas:
        clean: dict = {}
        for key, value in meta.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, list):
                clean[key] = " > ".join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        # ChromaDB rejects empty metadata dicts; pass None instead
        flat.append(clean if clean else None)
    return flat
