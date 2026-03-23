"""FastAPI application for the MARley manual evaluation UI.

Serves the evaluation web interface and provides a REST API for
loading evaluation items, saving human judgements, and tracking
progress. Run with::

    python -m evaluation.manual.app --items-dir data/testing/ --port 8000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from evaluation.manual.models import (
    Judgement,
    ManualJudgement,
    load_items,
    load_judgements,
    save_judgement,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _MODULE_DIR / "static"
_TEMPLATE_DIR = _MODULE_DIR / "templates"

app = FastAPI(title="MARley Manual Evaluation")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

# Runtime configuration — set by configure() before the server starts.
_items_dir: Path = Path("data/testing")
_judgements_dir: Path = Path("data/testing")


def configure(items_dir: str | Path, judgements_dir: str | Path | None = None) -> None:
    """Set the data directories for items and judgements."""
    global _items_dir, _judgements_dir
    _items_dir = Path(items_dir)
    _judgements_dir = Path(judgements_dir) if judgements_dir else _items_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_sources() -> list[dict]:
    """Find all manual-eval-items-*.json files and return source metadata."""
    sources = []
    for p in sorted(_items_dir.glob("manual-eval-items-*.json")):
        # Extract source name: manual-eval-items-generation-stpo.json → generation-stpo
        name = p.stem.replace("manual-eval-items-", "")
        sources.append({"name": name, "file": p.name, "path": str(p)})
    return sources


def _items_path(source: str) -> Path:
    return _items_dir / f"manual-eval-items-{source}.json"


def _judgements_path(source: str) -> Path:
    return _judgements_dir / f"manual-judgements-{source}.json"


# ---------------------------------------------------------------------------
# Pydantic models for request/response
# ---------------------------------------------------------------------------


class JudgementRequest(BaseModel):
    """Request body for POST /api/judgements."""
    item_id: str
    judgement: str
    notes: str = ""


class ProgressResponse(BaseModel):
    """Response for GET /api/progress."""
    total: int
    judged: int
    remaining: int
    by_category: dict[str, dict]


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the main evaluation UI page."""
    return templates.TemplateResponse("evaluate.html", {"request": request})


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.get("/api/sources")
async def list_sources():
    """List available evaluation item sources."""
    return _discover_sources()


@app.get("/api/items")
async def get_items(
    source: str,
    filter_kb: str | None = None,
    filter_distractors: int | None = None,
    filter_status: str | None = None,
    filter_category: str | None = None,
):
    """Return evaluation items with optional filters.

    Query parameters:
        source: Source name (required).
        filter_kb: Filter by knowledge_base metadata.
        filter_distractors: Filter by num_distractors metadata.
        filter_status: "pending" or "judged".
        filter_category: Filter by question category.
    """
    items_file = _items_path(source)
    if not items_file.exists():
        raise HTTPException(404, f"Source not found: {source}")

    items = load_items(items_file)
    judgements = load_judgements(_judgements_path(source))
    judged_ids = {j.item_id for j in judgements}
    judgement_map = {j.item_id: j for j in judgements}

    # Apply filters
    if filter_kb:
        items = [i for i in items if i.metadata.get("knowledge_base") == filter_kb]
    if filter_distractors is not None:
        items = [i for i in items if i.metadata.get("num_distractors") == filter_distractors]
    if filter_category:
        items = [i for i in items if i.category == filter_category]
    if filter_status == "pending":
        items = [i for i in items if i.id not in judged_ids]
    elif filter_status == "judged":
        items = [i for i in items if i.id in judged_ids]

    return [
        {
            "item": asdict(item),
            "judgement": asdict(judgement_map[item.id]) if item.id in judgement_map else None,
        }
        for item in items
    ]


@app.get("/api/items/{item_id}")
async def get_item(item_id: str, source: str):
    """Return a single evaluation item by ID."""
    items_file = _items_path(source)
    if not items_file.exists():
        raise HTTPException(404, f"Source not found: {source}")

    items = load_items(items_file)
    judgements = load_judgements(_judgements_path(source))
    judgement_map = {j.item_id: j for j in judgements}

    for item in items:
        if item.id == item_id:
            return {
                "item": asdict(item),
                "judgement": asdict(judgement_map[item.id]) if item.id in judgement_map else None,
            }

    raise HTTPException(404, f"Item not found: {item_id}")


@app.get("/api/progress")
async def get_progress(source: str):
    """Return progress statistics for a source."""
    items_file = _items_path(source)
    if not items_file.exists():
        raise HTTPException(404, f"Source not found: {source}")

    items = load_items(items_file)
    judgements = load_judgements(_judgements_path(source))
    judged_ids = {j.item_id for j in judgements}

    total = len(items)
    judged = sum(1 for i in items if i.id in judged_ids)

    # Group by category
    by_category: dict[str, dict] = {}
    for item in items:
        cat = item.category or "unknown"
        if cat not in by_category:
            by_category[cat] = {"total": 0, "judged": 0}
        by_category[cat]["total"] += 1
        if item.id in judged_ids:
            by_category[cat]["judged"] += 1

    return ProgressResponse(
        total=total,
        judged=judged,
        remaining=total - judged,
        by_category=by_category,
    )


@app.post("/api/judgements")
async def post_judgement(source: str, body: JudgementRequest):
    """Save a human judgement for an evaluation item."""
    # Validate judgement value
    try:
        judgement_value = Judgement(body.judgement)
    except ValueError:
        valid = [j.value for j in Judgement]
        raise HTTPException(400, f"Invalid judgement: {body.judgement}. Valid: {valid}")

    judgement = ManualJudgement(
        item_id=body.item_id,
        judgement=judgement_value,
        notes=body.notes,
    )

    save_judgement(judgement, _judgements_path(source))
    return {"status": "saved", "item_id": body.item_id, "judgement": body.judgement}


@app.get("/api/judgements")
async def get_judgements(source: str):
    """Return all judgements for a source."""
    judgements = load_judgements(_judgements_path(source))
    return [asdict(j) for j in judgements]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and start the evaluation server."""
    parser = argparse.ArgumentParser(description="MARley Manual Evaluation Server")
    parser.add_argument(
        "--items-dir",
        type=str,
        default="data/testing",
        help="Directory containing manual-eval-items-*.json files",
    )
    parser.add_argument(
        "--judgements-dir",
        type=str,
        default=None,
        help="Directory for saving judgements (default: same as items-dir)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )

    args = parser.parse_args()
    configure(args.items_dir, args.judgements_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
