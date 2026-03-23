# MARley Server

**Module:** `src/marley/server/`

The MARley server is a unified FastAPI application that exposes the complete pipeline (retrieval, abstention, generation) as a web service. Three interfaces are served under a single port:

1. **Production Chat UI** (`/`) — Minimal chat interface for students.
2. **Debug UI** (`/debug`) — Extended interface with configuration panel and retrieval details.
3. **Manual Evaluation UI** (`/evaluation`) — Phase 2 evaluation app mounted as a sub-application.

---

## Theoretical Background

Deploying a RAG pipeline as a web service introduces requirements beyond the core retrieval and generation logic. The server must manage expensive resources (embedding models, retriever indices, LLM connections) efficiently, handle concurrent requests, and provide configuration flexibility without restarting.

MARley uses **FastAPI** (Ramirez, 2019), an asynchronous Python web framework based on Starlette and Pydantic. FastAPI's automatic request validation (via Pydantic models) and OpenAPI documentation generation reduce boilerplate and ensure type safety at the API boundary.

The server implements a **lazy loading** strategy for retriever indices: indices are built on first use and cached for the server lifetime. This avoids the cost of loading all possible retriever-KB combinations at startup (which would require embedding multiple corpora). The cache key is a tuple of `(retriever_type, knowledge_bases, strategy)`, ensuring that different configurations do not interfere. This pattern is a form of the **memoization** optimization, applied at the resource-management level.

For the Fusion strategy, the server creates one retriever per knowledge base and wraps them in a `FusionRetriever`, reflecting the architectural separation between within-KB retrieval and cross-KB fusion.

---

## Architecture

### Unified Server

```
http://localhost:8000/
  |
  +-- /                     Production Chat UI (Jinja2 template)
  +-- /debug                Debug UI (Jinja2 template)
  +-- /evaluation           Manual Evaluation UI (sub-app mount)
  +-- /static/...           CSS, JS assets
  +-- /api/chat             POST  Pipeline query endpoint
  +-- /api/options          GET   Available configurations
  +-- /api/health           GET   Health check
  +-- /docs                 GET   Swagger API docs (auto)
```

### Startup Flow

1. Check Ollama connectivity — exit with error if unreachable.
2. Initialize `PipelineService` (no indices loaded yet).
3. Mount manual evaluation sub-app at `/evaluation`.
4. Mount static files, configure Jinja2 templates.
5. Start uvicorn.

### Lazy Loading

Retriever indices are created on first use and cached for the server lifetime:

- **Cache key:** `(retriever_type, frozenset(knowledge_bases), strategy)`
- **Single / Merged Pool:** One retriever, merged chunks from all specified KBs.
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
  --chunk-dir DIR        Directory containing chunk JSON files
  --eval-items-dir DIR   Directory for evaluation items
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
python -m evaluation.manual.app --port 8003
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
| `threshold` | float\|null | `null` | Abstention threshold override |

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

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | str | `"127.0.0.1"` | Server bind address |
| `port` | int | `8000` | Server port |
| `ollama_base_url` | str | `"http://localhost:11434"` | Ollama API URL |
| `ollama_model` | str | `"llama3.1:latest"` | LLM model |
| `chunk_dir` | Path | `data/chunks` | Chunk file directory |
| `k` | int | `5` | Default top-k |
| `default_retriever_type` | str | `"hybrid"` | Default retriever |
| `default_strategy` | str | `"merged_pool"` | Default strategy |
| `default_knowledge_bases` | list[str] | `["stpo", "faq-stpo", "faq-ao"]` | Default KBs |
| `evaluation_items_dir` | str | `"data/testing"` | Evaluation items dir |

## Module Structure

```
src/marley/server/
  __init__.py                 Module docstring
  __main__.py                 CLI entry: python -m src.marley.server
  app.py                      FastAPI application factory + routes + CLI
  config.py                   Server configuration, Ollama check
  models.py                   Pydantic request/response models
  service.py                  PipelineService (retriever cache, pipeline orchestration)
  templates/
    chat.html                 Production Chat UI
    debug.html                Debug UI
  static/
    chat.js                   Chat UI JavaScript
    debug.js                  Debug UI JavaScript
    style.css                 Shared stylesheet
```
