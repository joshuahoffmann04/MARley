from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from MARley.config import get_marley_config, get_marley_settings
from MARley.models import ChatRequest, ChatResponse, OptionsResponse
from MARley.service import MarleyPipelineService

settings = get_marley_settings()
runtime_config = get_marley_config(settings)
pipeline = MarleyPipelineService(settings=settings, runtime_config=runtime_config)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title=f"{settings.app_name} - Single App",
    description="Single-run MARley app with integrated retrieval, generation and frontend.",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    return pipeline.readiness()


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": settings.app_name},
    )


@app.get("/debug", response_class=HTMLResponse)
def debug_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="debug.html",
        context={"app_name": settings.app_name},
    )


@app.get("/api/options", response_model=OptionsResponse)
def options() -> OptionsResponse:
    return pipeline.options()


@app.get("/api/ready")
def api_ready() -> dict:
    return pipeline.readiness()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return pipeline.chat(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc
