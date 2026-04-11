# MARley Server

**Module:** `src/marley/server/`

The MARley server is a unified FastAPI application that exposes the complete pipeline (retrieval, abstention, generation) as a web service. Two interfaces are served under a single port:

1. **Production Chat UI** (`/`) — Minimal chat interface for students.
2. **Debug UI** (`/debug`) — Extended interface with configuration panel and retrieval details.

---

## Theoretical Background

Deploying a RAG pipeline as a web service introduces requirements beyond the core retrieval and generation logic. The server must manage expensive resources (embedding models, retriever indices, LLM connections) efficiently, handle concurrent requests, and provide configuration flexibility without restarting.

MARley uses **FastAPI** (Ramirez, 2019), an asynchronous Python web framework based on Starlette and Pydantic. FastAPI's automatic request validation (via Pydantic models) and OpenAPI documentation generation reduce boilerplate and ensure type safety at the API boundary.

The server implements a **lazy loading** strategy for retriever indices: indices are built on first use and cached for the server lifetime. This avoids the cost of loading all possible retriever-KB combinations at startup (which would require embedding multiple corpora). The cache key is a tuple of `(retriever_type, frozenset(knowledge_bases), strategy)`, ensuring that different configurations do not interfere. This pattern is a form of the **memoization** optimization, applied at the resource-management level.

For the Fusion strategy, the server creates one retriever per knowledge base and wraps them in a `FusionRetriever`, reflecting the architectural separation between within-KB retrieval and cross-KB fusion. For all other strategies (single, merged_pool), it wraps a single inner retriever in a `MergedRetriever`.

---

## Architecture

### Unified Server

```
http://localhost:8000/
  |
  +-- /                     Production Chat UI (Jinja2 template)
  +-- /debug                Debug UI (Jinja2 template)
  +-- /static/...           CSS, JS, images
  +-- /api/chat             POST  Pipeline query endpoint
  +-- /api/options          GET   Available configurations
  +-- /api/health           GET   Health check
  +-- /api/pdf/stpo         GET   Serve StPO PDF for in-browser viewer
  +-- /docs                 GET   Swagger API docs (auto)
```

### Startup Flow

1. Check Ollama connectivity — exit with error if unreachable.
2. Initialize `PipelineService` (no indices loaded yet).
3. Mount static files, configure Jinja2 templates.
4. Start uvicorn.

### Lazy Loading

Retriever indices are created on first use and cached for the server lifetime:

- **Cache key:** `(retriever_type, frozenset(knowledge_bases), strategy)`
- **Single / Merged Pool:** One inner retriever wrapped in `MergedRetriever`, merged chunks from all specified KBs.
- **Fusion:** One retriever per KB, wrapped in `FusionRetriever`.

## Startup

### CLI Arguments

```bash
python -m src.marley.server [OPTIONS]

Options:
  --host HOST            Server host (default: 127.0.0.1)
  --port PORT            Server port (default: 8000)
  --ollama-url URL       Ollama server URL (default: http://localhost:11434)
  --ollama-model MODEL   LLM model name (default: llama3.1:latest)
  --chunk-dir DIR        Directory containing chunk JSON files (default: data/chunks)
  --pdf-path PATH        Path to the StPO PDF for the in-browser viewer (default: data/raw/msc-computer-science.pdf)
  --mode {all,chat,debug}  Which UIs to serve (default: all)
```

### Ollama Requirement

The server checks Ollama connectivity at startup. If Ollama is not reachable, the server exits with a clear error message:

```
ERROR Ollama is not reachable at http://localhost:11434: [Errno 111] Connection refused
ERROR Start Ollama first, then restart the server.
```

### Individual Startup

Each interface can run independently under a different port:

```bash
python -m src.marley.server --mode chat --port 8001
python -m src.marley.server --mode debug --port 8002
```

## API Reference

### POST /api/chat

Process a question through the full pipeline.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | (required) | User question |
| `retriever_type` | string | `"hybrid"` | `"bm25"`, `"vector"`, or `"hybrid"` |
| `knowledge_bases` | list[str] | `["stpo", "faq-stpo", "faq-ao"]` | KBs to search |
| `strategy` | string | `"merged_pool"` | `"single"`, `"merged_pool"`, or `"fusion"` |
| `k` | int | `5` | Top-k retrieval count (1–50) |
| `threshold` | float\|null | `null` | Abstention threshold override (null = auto, based on normalization strategy) |

**Response body:**

| Field | Type | Description |
|---|---|---|
| `answer` | string | Generated answer text |
| `abstained` | bool | Whether the system abstained |
| `abstention_level` | int\|null | 1 = retrieval, 2 = LLM |
| `abstention_reason` | string | Reason for abstention |
| `confidence` | float | Top-1 normalized retrieval score |
| `sources` | list[SourceReference] | Retrieved chunks used |
| `config` | ChatConfigInfo | Configuration metadata |

### GET /api/health

Returns server and Ollama health status.

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` or `"degraded"` |
| `ollama` | string | `"connected"` or `"unavailable"` |
| `model` | string | Configured LLM model name |
| `cached_retrievers` | int | Number of cached retriever instances |
| `knowledge_bases` | list[str] | Available KBs (files found) |

### GET /api/options

Returns available pipeline configurations.

| Field | Type | Description |
|---|---|---|
| `retriever_types` | list[str] | Available retriever types |
| `knowledge_bases` | list[str] | Available knowledge bases |
| `strategies` | list[str] | Available combination strategies |
| `defaults` | dict | Default configuration values |
| `ollama_model` | string | Configured LLM model |
| `ollama_status` | string | `"connected"` or `"unavailable"` |

### GET /api/pdf/stpo

Serves the StPO PDF file for the in-browser viewer. Returns a `FileResponse` with `media_type="application/pdf"` and `content_disposition_type="inline"`. Returns 404 if the PDF path is not configured or the file does not exist.

## Chat UI

The production chat interface at `/` provides:

- Single-column layout (max-width 720px), centered.
- Textarea input with Enter to send (Shift+Enter for newline).
- User messages right-aligned, system messages left-aligned.
- Collapsible source references below each answer.
- Abstention messages with advisory office contact hint.
- Confidence badge (high/medium/low color coding).
- Loading spinner during generation.

## Debug UI

The debug interface at `/debug` provides:

- Two-column layout: settings sidebar (300px) + main content area.
- Configuration panel: retriever type, KB checkboxes, strategy, k, threshold.
- Server status indicator (Ollama connection, cached retrievers).
- Detailed results display: answer, confidence, configuration tags.
- Chunks table with chunk ID, score, and text preview.

## Configuration

### ServerConfig

Defaults shown below are simplified for readability. Mutable defaults (`chunk_dir`, `pdf_path`, `default_knowledge_bases`) use `field(default_factory=...)` in the actual code.

```python
@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    chunk_dir: Path = Path("data/chunks")                            # field(default_factory=...)
    pdf_path: Path | None = Path("data/raw/msc-computer-science.pdf")  # field(default_factory=...)
    k: int = 5
    default_retriever_type: str = "hybrid"
    default_strategy: str = "merged_pool"
    default_knowledge_bases: list[str] = ["stpo", "faq-stpo", "faq-ao"]  # field(default_factory=...)
```

### Chunk Path Mapping

The `CHUNK_PATHS` constant maps knowledge base identifiers to chunk file paths:

```python
CHUNK_PATHS = {
    "stpo": "data/chunks/stpo-chunks.json",
    "faq-stpo": "data/chunks/faq-stpo-chunks.json",
    "faq-ao": "data/chunks/faq-ao-chunks.json",
}
```

### Ollama Check

`check_ollama(base_url, timeout=5.0)` sends a GET request to `{base_url}/api/tags` and returns a dict with `available` (bool) and optional `error` (str).

## Pipeline Orchestration

The `run_with_abstention()` function in `src/marley/server/pipeline.py` orchestrates the full abstention-aware pipeline. See [abstention.md](abstention.md) for details on the two-level abstention mechanism.

```python
def run_with_abstention(
    query: str,
    retriever: Retriever,
    generator: Generator,
    *,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    normalization_strategy: str = "vector",
    normalization_params: dict[str, Any] | None = None,
) -> AbstentionResult:
```

## PipelineService

`PipelineService` manages the pipeline lifecycle:

- **Constructor:** Takes a `ServerConfig`, creates an `OllamaGenerator`.
- **`get_retriever(retriever_type, knowledge_bases, strategy, k_rrf)`**: Returns a cached or newly created retriever.
- **`chat(query, ...)`**: Full pipeline call returning a structured dict with answer, sources, confidence, abstention info, and config metadata. Sources include `start_page`, `end_page`, and `section_title` metadata for PDF viewer integration. Source text is truncated to `SOURCE_TEXT_TRUNCATION` (500) characters.
- **`available_knowledge_bases()`**: Returns KBs for which chunk files exist on disk.
- **`cached_retriever_count`**: Number of cached retriever instances (property).
- **`generator_model`**: Configured LLM model name (property).

## Pydantic Models (`models.py`)

| Model | Purpose |
|---|---|
| `ChatRequest` | Request body for POST /api/chat |
| `ChatResponse` | Response body for POST /api/chat |
| `SourceReference` | A single source chunk in the response |
| `ChatConfigInfo` | Configuration metadata in the response |
| `OptionsResponse` | Response for GET /api/options |
| `HealthResponse` | Response for GET /api/health |

## Module Structure

```
src/marley/server/
  __init__.py                 Exports: PipelineService, run_with_abstention
  __main__.py                 CLI entry: python -m src.marley.server
  app.py                      FastAPI application factory (create_app) + routes + CLI (main)
  config.py                   ServerConfig, CHUNK_PATHS, check_ollama
  models.py                   Pydantic request/response models
  pipeline.py                 run_with_abstention() orchestrator
  service.py                  PipelineService (retriever cache, pipeline orchestration)
  templates/
    chat.html                 Production Chat UI
    debug.html                Debug UI
  static/
    chat.js                   Chat UI JavaScript
    debug.js                  Debug UI JavaScript
    logo-uni-marburg.png      Philipps-Uni Marburg logo
    style.css                 Shared stylesheet
```
