from __future__ import annotations

from fastapi import FastAPI, HTTPException

from generator.config import get_generator_config, get_generator_settings
from generator.models import GenerateRequest, GenerateResponse
from generator.service import GeneratorBackendUnavailableError, GeneratorService

settings = get_generator_settings()
runtime_config = get_generator_config(settings)
service = GeneratorService(settings=settings, runtime_config=runtime_config)

app = FastAPI(
    title=f"{settings.app_name} - Generator",
    description="Answer generation service for MARley.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    return service.readiness()


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        return service.generate(request)
    except GeneratorBackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
