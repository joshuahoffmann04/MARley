from __future__ import annotations

from fastapi import FastAPI, HTTPException

from retrieval.sparse_retrieval.config import (
    get_sparse_retrieval_config,
    get_sparse_retrieval_settings,
)
from retrieval.sparse_retrieval.models import (
    IndexRebuildRequest,
    IndexRebuildResponse,
    SearchRequest,
    SearchResponse,
)
from retrieval.sparse_retrieval.service import SparseBM25Retriever

settings = get_sparse_retrieval_settings()
runtime_config = get_sparse_retrieval_config(settings)
retriever = SparseBM25Retriever(settings=settings, runtime_config=runtime_config)

app = FastAPI(
    title=f"{settings.app_name} - Sparse Retrieval (BM25)",
    description="Sparse retrieval service for MARley.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    return retriever.readiness()


@app.post("/index/rebuild", response_model=IndexRebuildResponse)
def rebuild_index(request: IndexRebuildRequest) -> IndexRebuildResponse:
    document_id = request.document_id or settings.document_id
    try:
        index_stats, quality_flags = retriever.rebuild_index(document_id=document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}") from exc

    return IndexRebuildResponse(
        document_id=document_id,
        index_stats=index_stats,
        quality_flags=quality_flags,
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    document_id = request.document_id or settings.document_id
    try:
        return retriever.search(
            query=request.query,
            document_id=document_id,
            top_k=request.top_k,
            source_types=request.source_types,
            rebuild_if_stale=request.rebuild_if_stale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
