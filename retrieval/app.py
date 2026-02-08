from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from MARley.config import get_settings
from pdf_extractor.config import get_pdf_extractor_config
from pdf_extractor.models import ExtractionJob, ExtractionRequest, ExtractionResponse
from pdf_extractor.service import resolve_source_file, run_extraction

settings = get_settings()
runtime_config = get_pdf_extractor_config(settings)

app = FastAPI(
    title=f"{settings.app_name} - PDF Extractor",
    version="0.1.0",
    description="PDF extraction service for MARley.",
)

_JOBS: dict[str, ExtractionJob] = {}
_CACHE: dict[str, dict] = {}
_LOCK = Lock()


def _resolve_output_dir(raw_value: str | None) -> Path:
    if not raw_value:
        return runtime_config.knowledgebases_dir.resolve()

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return candidate.resolve()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/v1/extractions", response_model=ExtractionResponse)
def create_extraction(request: ExtractionRequest) -> ExtractionResponse:
    document_id = request.document_id or settings.document_id
    job_id = str(uuid4())
    started_at = datetime.now(timezone.utc)

    try:
        source_file = resolve_source_file(request.source_file, document_id=document_id)
        output_dir = _resolve_output_dir(request.output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = ExtractionJob(
        job_id=job_id,
        status="running",
        document_id=document_id,
        source_file=str(source_file),
        started_at=started_at,
    )
    with _LOCK:
        _JOBS[job_id] = job

    try:
        result, output_file = run_extraction(
            document_id=document_id,
            source_file=source_file,
            output_dir=output_dir,
            runtime_config=runtime_config,
        )
        finished_at = datetime.now(timezone.utc)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        completed_job = ExtractionJob(
            job_id=job_id,
            status="completed",
            document_id=document_id,
            source_file=str(source_file),
            output_file=str(output_file),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            quality_flags=result.quality_flags,
            section_count=len(result.sections),
            table_count=len(result.tables),
        )

        with _LOCK:
            _JOBS[job_id] = completed_job
            _CACHE[job_id] = result.model_dump(mode="json")

        return ExtractionResponse(job=completed_job)
    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)
        failed_job = ExtractionJob(
            job_id=job_id,
            status="failed",
            document_id=document_id,
            source_file=str(source_file),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            error=str(exc),
        )
        with _LOCK:
            _JOBS[job_id] = failed_job
        raise HTTPException(status_code=500, detail=f"Extraktion fehlgeschlagen: {exc}") from exc


@app.get("/v1/extractions/{job_id}", response_model=ExtractionResponse)
def get_extraction_status(job_id: str) -> ExtractionResponse:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} nicht gefunden.")
    return ExtractionResponse(job=job)


@app.get("/v1/extractions/{job_id}/result")
def get_extraction_result(job_id: str) -> JSONResponse:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} nicht gefunden.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job {job_id} ist noch nicht abgeschlossen.")
    if not job.output_file:
        raise HTTPException(status_code=500, detail=f"Job {job_id} enthält keinen Ergebnis-Pfad.")

    cached = _CACHE.get(job_id)
    if cached is not None:
        return JSONResponse(content=cached)

    output_file = Path(job.output_file)
    if not output_file.exists():
        raise HTTPException(status_code=500, detail=f"Ergebnisdatei fehlt: {output_file}")

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    with _LOCK:
        _CACHE[job_id] = payload
    return JSONResponse(content=payload)
