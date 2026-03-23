"""FastAPI application for the MARley pipeline server.

Serves the production chat UI, debug UI, and manual evaluation UI
under a single application. Run with::

    python -m src.marley.server --port 8000

Or start individual components separately::

    python -m src.marley.server --mode chat --port 8001
    python -m src.marley.server --mode debug --port 8002
    python -m evaluation.manual.app --port 8003
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.marley.server.config import (
    RETRIEVER_TYPES,
    STRATEGIES,
    ServerConfig,
    check_ollama,
)
from src.marley.server.models import (
    ChatConfigInfo,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    OptionsResponse,
    SourceReference,
)
from src.marley.server.service import PipelineService

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _MODULE_DIR / "static"
_TEMPLATE_DIR = _MODULE_DIR / "templates"


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Server configuration (default: ServerConfig()).

    Returns:
        Configured FastAPI application.
    """
    config = config or ServerConfig()
    service = PipelineService(config)

    app = FastAPI(
        title="MARley",
        description="MARburg Study Advising ChatBot",
    )

    # Static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

    # Mount manual evaluation sub-app
    try:
        from evaluation.manual.app import app as eval_app
        from evaluation.manual.app import configure as eval_configure

        eval_configure(config.evaluation_items_dir)
        app.mount("/evaluation", eval_app)
        logger.info("Manual evaluation UI mounted at /evaluation")
    except ImportError:
        logger.warning("evaluation.manual.app not available; /evaluation disabled")

    # Redirect /evaluation (no trailing slash) to /evaluation/ so that
    # the sub-app's relative static/API paths resolve correctly.
    @app.get("/evaluation")
    async def evaluation_redirect() -> RedirectResponse:
        return RedirectResponse(url="/evaluation/", status_code=307)

    # --- UI Routes ---

    @app.get("/", response_class=HTMLResponse)
    async def chat_page(request: Request) -> HTMLResponse:
        """Serve the production chat UI."""
        return templates.TemplateResponse(request, "chat.html")

    @app.get("/debug", response_class=HTMLResponse)
    async def debug_page(request: Request) -> HTMLResponse:
        """Serve the debug UI."""
        return templates.TemplateResponse(request, "debug.html")

    # --- API Routes ---

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return server and Ollama health status."""
        ollama = check_ollama(config.ollama_base_url)
        return HealthResponse(
            status="ok" if ollama["available"] else "degraded",
            ollama="connected" if ollama["available"] else "unavailable",
            model=config.ollama_model,
            cached_retrievers=service.cached_retriever_count,
            knowledge_bases=service.available_knowledge_bases(),
        )

    @app.get("/api/options", response_model=OptionsResponse)
    async def options() -> OptionsResponse:
        """Return available pipeline configurations."""
        ollama = check_ollama(config.ollama_base_url)
        return OptionsResponse(
            retriever_types=RETRIEVER_TYPES,
            knowledge_bases=service.available_knowledge_bases(),
            strategies=STRATEGIES,
            defaults={
                "retriever_type": config.default_retriever_type,
                "knowledge_bases": config.default_knowledge_bases,
                "strategy": config.default_strategy,
                "k": config.k,
            },
            ollama_model=config.ollama_model,
            ollama_status="connected" if ollama["available"] else "unavailable",
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        """Process a chat question through the full pipeline."""
        if not body.query.strip():
            raise HTTPException(400, "Query must not be empty.")

        try:
            result = service.chat(
                body.query.strip(),
                retriever_type=body.retriever_type,
                knowledge_bases=body.knowledge_bases,
                strategy=body.strategy,
                k=body.k,
                threshold=body.threshold,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(500, f"Pipeline error: {exc}") from exc

        return ChatResponse(
            answer=result["answer"],
            abstained=result["abstained"],
            abstention_level=result["abstention_level"],
            abstention_reason=result["abstention_reason"],
            confidence=result["confidence"],
            sources=[SourceReference(**s) for s in result["sources"]],
            config=ChatConfigInfo(**result["config"]),
        )

    return app


# --- CLI Entry Point ---


def main() -> None:
    """Parse arguments and start the MARley server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="MARley Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="llama3.1:latest")
    parser.add_argument("--chunk-dir", default="data/chunks")
    parser.add_argument("--eval-items-dir", default="data/testing")
    parser.add_argument(
        "--mode",
        choices=["all", "chat", "debug"],
        default="all",
        help="Which UIs to serve (default: all)",
    )

    args = parser.parse_args()

    # Check Ollama connectivity before starting
    ollama_status = check_ollama(args.ollama_url)
    if not ollama_status["available"]:
        logger.error(
            "Ollama is not reachable at %s: %s",
            args.ollama_url,
            ollama_status.get("error", "unknown"),
        )
        logger.error("Start Ollama first, then restart the server.")
        sys.exit(1)

    logger.info(
        "Ollama connected at %s (model: %s)",
        args.ollama_url,
        args.ollama_model,
    )

    config = ServerConfig(
        host=args.host,
        port=args.port,
        ollama_base_url=args.ollama_url,
        ollama_model=args.ollama_model,
        chunk_dir=Path(args.chunk_dir),
        evaluation_items_dir=args.eval_items_dir,
    )

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
