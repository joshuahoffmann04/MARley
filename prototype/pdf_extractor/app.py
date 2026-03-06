from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from pdf_extractor.config import get_pdf_extractor_config, get_pdf_extractor_settings
from pdf_extractor.models import ExtractionRequest, ExtractionResponse
from pdf_extractor.service import resolve_source_file, run_extraction

settings = get_pdf_extractor_settings()
runtime_config = get_pdf_extractor_config(settings)

app = FastAPI(
    title=f"{settings.app_name} - PDF Extractor",
    description="PDF extraction service for MARley.",
)


def _resolve_output_dir(raw_value: str | None) -> Path:
    if not raw_value:
        return runtime_config.knowledgebases_dir.resolve()

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (settings.project_root / candidate).resolve()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/extractions", response_model=ExtractionResponse)
def create_extraction(request: ExtractionRequest) -> ExtractionResponse:
    document_id = request.document_id or settings.document_id

    try:
        source_file = resolve_source_file(
            request_source_file=request.source_file,
            document_id=document_id,
            settings=settings,
        )
        output_dir = _resolve_output_dir(request.output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result, output_file = run_extraction(
            document_id=document_id,
            source_file=source_file,
            output_dir=output_dir,
            runtime_config=runtime_config,
            persist_result=request.persist_result,
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    return ExtractionResponse(
        document_id=document_id,
        source_file=str(source_file),
        output_file=str(output_file) if output_file else None,
        result=result,
    )
