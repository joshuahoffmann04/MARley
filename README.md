# MARley

**MARburg Study Advising ChatBot** — A Retrieval-Augmented Generation (RAG) pipeline for answering questions about the M.Sc. Computer Science program at Philipps-Universitat Marburg.

Bachelor thesis by Joshua Hoffmann, Department of Mathematics and Computer Science, Philipps-Universitat Marburg.

---

## Pipeline

```
PDF / FAQ Data
     |
     +-- 1. Extractor      Extract sections and tables from the StPO PDF
     +-- 2. Chunker         Split into retrieval-ready chunks (text, table, FAQ)
     +-- 3. Retrieval       Find relevant chunks for a query (BM25 / Vector / Hybrid / Fusion)
     +-- 4. Generation      Generate an answer from the retrieved context (Ollama LLM)
     +-- 4.5 Abstention     Two-level controlled abstention (retrieval confidence + LLM)
     +-- 5. Server          Chat interface for students (Philipps-Uni Marburg design)
```

**Implemented:** Stages 1–5 + Evaluation harness (retrieval + generation + abstention + end-to-end).

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **NVIDIA GPU with CUDA 12.1 driver** (16 GB VRAM recommended; used by the embedding model and RAGAS evaluator)
- **Ollama** (local LLM server, required for generation and the chat server)

### Installation

Install PyTorch with CUDA support first, then the rest of the dependencies:

```bash
git clone https://github.com/joshuahoffmann04/MARley.git
cd MARley

# 1. CUDA-enabled PyTorch (required)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Project + remaining dependencies
pip install -e .
```

Verify the GPU is picked up:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

### Configuration

Copy `.env.example` to `.env` and fill in any secrets you need. The only
supported variable is `OPENAI_API_KEY`, which is required when running the
generation evaluation with the OpenAI judge (`--judge openai`); the default
Ollama judge needs no configuration.

```bash
cp .env.example .env
# then edit .env and paste your sk-... key if you plan to use --judge openai
```

### Ollama setup

Set `OLLAMA_NUM_PARALLEL=2` before starting the server. Two parallel slots
are the sweet spot for an 8B model on a 16 GB GPU: more causes VRAM
contention, fewer leaves GPU cycles idle. The RAGAS judge batch size of
20 (= 2 × 10 waves) is tuned to match this setting.

Ollama reads the variable **only at startup**, so any running instance
must be restarted after setting it. On Windows the cleanest flow is:

```powershell
# 1. Stop the tray app + server
Stop-Process -Name 'ollama','ollama app' -Force -ErrorAction SilentlyContinue

# 2. Set the variable for this shell + start the server
$env:OLLAMA_NUM_PARALLEL = "2"
ollama serve
```

To make the setting survive reboots, add `OLLAMA_NUM_PARALLEL=2` as a
user environment variable (Windows: "Edit environment variables for
your account") and restart Ollama once.

On Linux/macOS:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

### Data Preparation

```bash
# 1. Extract sections and tables from the StPO PDF
python -c "from src.marley.extractor import extract, save; save(extract())"

# 2. Chunk the extracted data (StPO + FAQ knowledge bases)
python -c "
from src.marley.chunker import chunk_stpo, save, chunk_faq, save_faq
save(chunk_stpo(), 'data/chunks/stpo-chunks.json')
save_faq(chunk_faq('data/knowledgebase/faq-stpo.json'), 'data/chunks/faq-stpo-chunks.json')
"
```

### Startup checklist

The full cold-start sequence for running evaluation or the chat server:

1. **Ollama with 2 parallel slots** — see the "Ollama setup" section
   above. Verify with `curl http://localhost:11434/api/tags`; the model
   list should include `llama3.1:latest`.
2. **GPU + venv sanity check** (optional but fast):
   ```bash
   python -c "import torch; assert torch.cuda.is_available()"
   ```
3. **For evaluation only** — validate data and environment in one go:
   ```bash
   python -m evaluation --check
   ```
   This confirms the GPU is detected, all eval/chunk files are present,
   and Ollama is reachable.
4. **Start whichever you need**:
   - Chat server: `python -m src.marley.server --port 8000`
   - Evaluation: one of the `python -m evaluation ...` commands in the
     ["Run Evaluation"](#run-evaluation) section below.

### Start the Chat Server

```bash
# Start Ollama with 2 parallel slots (see Ollama setup)
$env:OLLAMA_NUM_PARALLEL = "2"
ollama serve

# In a second terminal: start MARley
python -m src.marley.server --port 8000
```

Open http://localhost:8000 in your browser.

### Run Tests

```bash
# Run all tests (674 tests)
python -m pytest tests/ evaluation/tests/

# Unit tests only (fast, no data files or services needed)
python -m pytest -m "not integration"

# Integration tests only (requires data files)
python -m pytest -m integration
```

### Run Evaluation

```bash
python -m evaluation --check                        # Validate data requirements
python -m evaluation --retrieval                    # Retrieval metrics (Precision@k, Recall@k, MRR)
python -m evaluation --rrf-tuning                   # Sweep k_rrf for Hybrid and Fusion
python -m evaluation --generation                   # Generation quality (RAGAS, Ollama judge)
python -m evaluation --generation --judge openai    # Same, judged by OpenAI gpt-4o-mini
python -m evaluation --abstention                   # Abstention metrics (precision, recall, F1)
python -m evaluation --e2e                          # End-to-end (33 configs x 100 questions, RAGAS-scored)
python -m evaluation --e2e --judge openai           # Same, OpenAI judge for answer scoring
python -m evaluation --all --judge openai           # Full run, OpenAI judge for generation + E2E
```

**Judge backends.** The generator under evaluation is always Ollama. The
RAGAS judge that scores its answers is swappable via `--judge`:

| Backend | Model | Batch size | Requires |
|---|---|---|---|
| `ollama` (default) | Your local Ollama model | 20 | `OLLAMA_NUM_PARALLEL=2` |
| `openai` | `gpt-4o-mini` | 50 | `OPENAI_API_KEY` in `.env` |

The `--judge` flag affects every step that scores free-form answers with
RAGAS — `--generation` and `--e2e` — and, through `--all`, both of those
phases at once. Abstention, retrieval, and RRF tuning measure
deterministic booleans and set operations; they ignore the flag.

In E2E, RAGAS is applied only to samples where the system answered an
answerable question (Faithfulness, Answer Relevancy, Factual
Correctness). Abstained results and hallucinations on unanswerable
questions keep `NaN` on those fields and are excluded from the averages.

For quick iteration on the generation step, restrict the sweep with
`--subset` and `--distractor-levels`:

```bash
python -m evaluation --generation --subset 10 --distractor-levels 0,5,10 --judge openai
```

---

## Project Structure

```
src/marley/
+-- models/          Shared layer: data classes, ABCs, constants, scoring utilities
+-- extractor/       PDF extraction (PyMuPDF + pdfplumber)
+-- chunker/         PDF chunking (sentence-aligned sliding window) + FAQ chunking
+-- retrieval/       BM25, Vector, Hybrid (RRF), Fusion, and Merged retrievers
+-- generator/       Ollama LLM backend for answer generation
+-- abstention/      Two-level abstention (retrieval confidence + LLM detection)
+-- server/          Pipeline orchestrator + FastAPI server (chat UI, debug UI, API)

evaluation/
+-- retrieval/       Retrieval evaluation (Precision@k, Recall@k, MRR) + RRF tuning
+-- generation/      Generation evaluation (RAGAS: faithfulness, answer relevancy)
+-- abstention/      Abstention evaluation (precision, recall, F1, threshold sweep)
+-- end_to_end/      End-to-end pipeline evaluation (33 configs x 100 questions)
+-- tests/           Evaluation unit tests
+-- validate.py      Data requirement validation
+-- __main__.py      Unified evaluation CLI (python -m evaluation)

tests/               Unit and integration tests (mirrored by component)

data/
+-- raw/             Source PDFs
+-- knowledgebase/   Extracted data + FAQ knowledge bases
+-- chunks/          Chunked output (retrieval-ready JSON + ChromaDB stores)
+-- evaluation/      Evaluation datasets and results

docs/
+-- source/          Source code documentation (per component)
+-- testing/         Test documentation (per component)
+-- evaluation/      Evaluation documentation + results
+-- evaluation-testing/  Evaluation test documentation
+-- uml/             UML class diagram
```

---

## Knowledge Bases

| Knowledge Base | Source | Chunks | Description |
|---|---|---|---|
| **stpo** | `msc-computer-science.pdf` | 153 | Study and examination regulations (text + tables) |
| **faq-stpo** | `faq-stpo.json` | 1039 | Synthetic FAQ derived from the StPO |
| **faq-ao** | `faq-ao.json` | 0 | Student questions answered by the advisory office (placeholder) |

---

## Tests

**674 tests** (672 passed, 2 skipped) across source code and evaluation.

### Source Tests (447)

| Component | Tests | Test Files |
|---|---|---|
| Extractor | 81 | `tests/extractor/test_extractor.py` |
| PDF Chunker | 60 | `tests/chunker/test_pdf_chunker.py` |
| FAQ Chunker | 35 | `tests/chunker/test_faq_chunker.py` |
| BM25 Retrieval | 27 | `tests/retrieval/test_bm25.py` |
| Vector Retrieval | 25 | `tests/retrieval/test_vector.py` |
| Hybrid Retrieval | 26 | `tests/retrieval/test_hybrid.py` |
| Fusion Retrieval | 37 | `tests/retrieval/test_fusion.py` |
| Merged Retrieval | 14 | `tests/retrieval/test_merged.py` |
| Generator | 24 | `tests/generator/test_generator.py` |
| Data Models | 34 | `tests/models/test_models.py` |
| Score Normalization | 20 | `tests/models/test_scoring.py` |
| Abstention Detection | 12 | `tests/abstention/test_detection.py` |
| Server API | 17 | `tests/server/test_api.py` |
| Server Models | 15 | `tests/server/test_models.py` |
| Server Pipeline | 10 | `tests/server/test_pipeline.py` |
| Server Service | 10 | `tests/server/test_service.py` |

### Evaluation Tests (227)

| Component | Tests | Test Files |
|---|---|---|
| Retrieval Metrics | 33 | `evaluation/tests/retrieval/test_metrics.py` |
| Retrieval Combined | 25 | `evaluation/tests/retrieval/test_combined.py` |
| Retrieval Evaluate | 13 | `evaluation/tests/retrieval/test_evaluate.py` |
| RRF Tuning | 10 | `evaluation/tests/retrieval/test_rrf_tuning.py` |
| Generation Metrics | 15 | `evaluation/tests/generation/test_metrics.py` |
| Generation Evaluate | 19 | `evaluation/tests/generation/test_evaluate.py` |
| Generation Combined | 15 | `evaluation/tests/generation/test_combined.py` |
| Judge Factory | 8 | `evaluation/tests/test_judge.py` |
| Abstention Metrics | 10 | `evaluation/tests/abstention/test_metrics.py` |
| Abstention Evaluate | 12 | `evaluation/tests/abstention/test_evaluate.py` |
| E2E Config | 10 | `evaluation/tests/end_to_end/test_config.py` |
| E2E Evaluate | 22 | `evaluation/tests/end_to_end/test_evaluate.py` |
| E2E Metrics | 19 | `evaluation/tests/end_to_end/test_metrics.py` |
| Utilities | 16 | `evaluation/tests/test_utils.py` |

Integration tests (`@pytest.mark.integration`) require data files or external services and are skipped when using `-m "not integration"`.

---

## Test Coverage

Overall coverage: **87.1%** (3950 statements, 509 missed).

| Component | Coverage | Notes |
|---|---|---|
| Models | 100% | All data classes, constants, scoring |
| Retrieval | 96–100% | BM25, Hybrid, Fusion, Merged: 100%; Vector: 96.5% |
| Extractor | 97.1% | Edge cases in PDF parsing |
| Chunker | 89–99% | PDF chunker: 89.4%, FAQ chunker: 99.0% |
| Generator | 100% | Prompt, base, Ollama adapter |
| Abstention | 100% | Detection module |
| Server | 65–100% | App routes: 65.4% (requires running server), rest: 73–100% |
| Evaluation | 82–100% | CLI entry point: 0% (requires Ollama); modules: 82–100% |

Conscious gaps: Server app routes and the evaluation CLI require a running Ollama server and are tested via integration tests. The `__main__.py` entry point (0%) is the CLI dispatcher, tested end-to-end.

---

## Documentation

### Source Code Documentation

| Document | Path |
|---|---|
| Extractor | `docs/source/extractor.md` |
| Chunker | `docs/source/chunker.md` |
| Retrieval Overview | `docs/source/retrieval/overview.md` |
| BM25 Retrieval | `docs/source/retrieval/bm25.md` |
| Vector Retrieval | `docs/source/retrieval/vector.md` |
| Hybrid Retrieval | `docs/source/retrieval/hybrid.md` |
| Fusion Retrieval | `docs/source/retrieval/fusion.md` |
| Merged Retrieval | `docs/source/retrieval/merged.md` |
| Generator | `docs/source/generator.md` |
| Abstention | `docs/source/abstention.md` |
| Models | `docs/source/models.md` |
| Data Structures | `docs/source/data.md` |
| Server | `docs/source/server.md` |

### Test Documentation

| Document | Path |
|---|---|
| Overview | `docs/testing/overview.md` |
| Extractor Tests | `docs/testing/extractor.md` |
| Chunker Tests | `docs/testing/chunker.md` |
| Retrieval Tests | `docs/testing/retrieval.md` |
| Generator Tests | `docs/testing/generator.md` |
| Models Tests | `docs/testing/models.md` |
| Abstention Tests | `docs/testing/abstention.md` |
| Server Tests | `docs/testing/server.md` |

### Evaluation Documentation

| Document | Path |
|---|---|
| Overview | `docs/evaluation/overview.md` |
| Retrieval Evaluation | `docs/evaluation/retrieval.md` |
| Generation Evaluation | `docs/evaluation/generation.md` |
| Abstention Evaluation | `docs/evaluation/abstention.md` |
| End-to-End Evaluation | `docs/evaluation/end-to-end.md` |
| Results Overview | `docs/evaluation/results/README.md` |

### Evaluation Test Documentation

| Document | Path |
|---|---|
| Overview | `docs/evaluation-testing/overview.md` |
| Retrieval Tests | `docs/evaluation-testing/retrieval.md` |
| Generation Tests | `docs/evaluation-testing/generation.md` |
| Abstention Tests | `docs/evaluation-testing/abstention.md` |
| End-to-End Tests | `docs/evaluation-testing/end-to-end.md` |
| Utility Tests | `docs/evaluation-testing/utils.md` |

---

## Dependencies

### Core

| Package | Purpose |
|---|---|
| PyMuPDF | PDF text and metadata extraction |
| pdfplumber | PDF table extraction |
| syntok | Sentence segmentation |
| tiktoken | Token counting (cl100k_base) |
| rank-bm25 | Sparse keyword retrieval |
| sentence-transformers | Dense embedding generation (all-mpnet-base-v2) |
| chromadb | Vector store (embedded, persistent) |
| ollama | Local LLM generation via Ollama HTTP API |
| FastAPI | Web framework for the chat server |
| uvicorn | ASGI server |
| Jinja2 | HTML templating |

### Evaluation (optional)

| Package | Purpose |
|---|---|
| ragas | RAG evaluation metrics (faithfulness, answer relevancy) |
| openai | OpenAI API client (used by the configurable RAGAS judge) |
| python-dotenv | Loads `OPENAI_API_KEY` from `.env` into the CLI environment |

### Development

| Package | Purpose |
|---|---|
| pytest | Test framework |
| pytest-cov | Coverage reporting |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Joshua Hoffmann — Bachelor thesis, Department of Mathematics and Computer Science, Philipps-Universitat Marburg (2026).
