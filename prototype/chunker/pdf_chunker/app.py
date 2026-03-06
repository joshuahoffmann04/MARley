from __future__ import annotations

from fastapi import FastAPI, HTTPException

from chunker.pdf_chunker.config import get_pdf_chunker_config, get_pdf_chunker_settings
from chunker.pdf_chunker.models import ChunkingRequest, ChunkingResponse
from chunker.pdf_chunker.service import run_pdf_chunking

settings = get_pdf_chunker_settings()
runtime_config = get_pdf_chunker_config(settings)

app = FastAPI(
    title=f"{settings.app_name} - PDF Chunker",
    description="PDF chunking service for MARley.",
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
        result, input_file, output_file = run_pdf_chunking(
            document_id=document_id,
            request_input_file=request.input_file,
            request_output_dir=request.output_dir,
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
        input_file=str(input_file),
        output_file=str(output_file) if output_file else None,
        result=result,
    )
