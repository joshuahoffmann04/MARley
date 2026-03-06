from __future__ import annotations

from fastapi import FastAPI, HTTPException

from chunker.faq_chunker.config import get_faq_chunker_config, get_faq_chunker_settings
from chunker.faq_chunker.models import ChunkingRequest, ChunkingResponse
from chunker.faq_chunker.service import run_faq_chunking

settings = get_faq_chunker_settings()
runtime_config = get_faq_chunker_config(settings)

app = FastAPI(
    title=f"{settings.app_name} - FAQ Chunker",
    description="FAQ chunking service for MARley.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/chunks", response_model=ChunkingResponse)
def create_chunks(request: ChunkingRequest) -> ChunkingResponse:
    document_id = request.document_id or settings.document_id
    try:
        result, input_file, output_file = run_faq_chunking(
            document_id=document_id,
            request_input_file=request.input_file,
            request_output_dir=request.output_dir,
            request_faq_source=request.faq_source,
            persist_result=request.persist_result,
            settings=settings,
            runtime_config=runtime_config,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}") from exc

    return ChunkingResponse(
        document_id=document_id,
        faq_source=result.faq_source,
        input_file=str(input_file),
        output_file=str(output_file) if output_file else None,
        result=result,
    )
