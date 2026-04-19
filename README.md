# MARley

**MAR**ley: A RAG-Based Chatbot for Answering Questions About University Study Regulations

> Bachelor thesis, Department of Mathematics and Computer Science,
> Philipps-Universität Marburg, 2026 · Joshua Hoffmann

---

## Overview

MARley ingests the official study and examination regulations (StPO) and a
curated FAQ corpus, retrieves relevant passages for a student question, and
generates an answer with a local LLM. A two-level abstention mechanism
keeps the bot from fabricating answers outside its knowledge base. Every
stage is independently benchmarked and the full 33-configuration matrix
is evaluated end-to-end with a configurable RAGAS judge.

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Chat Server](#running-the-chat-server)
6. [Running the Evaluation](#running-the-evaluation)
7. [Data Preparation](#data-preparation)
8. [Project Layout](#project-layout)
9. [Knowledge Bases](#knowledge-bases)
10. [Testing](#testing)
11. [Documentation](#documentation)
12. [License and Author](#license-and-author)

---

## Architecture

```
PDF / FAQ sources
        │
        ├─ 1. Extractor   PyMuPDF + pdfplumber — sections and tables
        ├─ 2. Chunker     Sentence-aligned windows (PDF) + FAQ pairs
        ├─ 3. Retrieval   BM25 · Vector · Hybrid (RRF) · Fusion · Merged
        ├─ 4. Generation  Ollama LLM (llama3.1:latest, 8B, Q4_K_M)
        ├─ 4.5 Abstention Level 1 (retrieval confidence) + Level 2 (LLM)
        └─ 5. Server      FastAPI chat UI in Philipps-Uni Marburg design
```

| Component  | Module                   | Notes                                 |
| ---------- | ------------------------ | ------------------------------------- |
| Extractor  | `src/marley/extractor/`  | PDF → sections, tables, text          |
| Chunker    | `src/marley/chunker/`    | PDF + FAQ chunkers, sliding window    |
| Retrieval  | `src/marley/retrieval/`  | Five strategies, ChromaDB for vectors |
| Generator  | `src/marley/generator/`  | Ollama HTTP backend + prompt template |
| Abstention | `src/marley/abstention/` | Two-level detection module            |
| Server     | `src/marley/server/`     | FastAPI pipeline + chat UI            |
| Evaluation | `evaluation/`            | Unified CLI for all eval steps        |

---

## Prerequisites

| Resource | Requirement                                          |
| -------- | ---------------------------------------------------- |
| OS       | Windows 10/11, Linux, or macOS                       |
| Python   | 3.12 or newer                                        |
| GPU      | NVIDIA with CUDA 12.1 driver, 16 GB VRAM recommended |
| Ollama   | Installed and reachable on `http://localhost:11434`  |
| Disk     | ~4 GB for models and ChromaDB stores                 |

The pipeline treats CUDA as baseline — there is no CPU fallback for
embeddings or RAGAS. A missing GPU causes an early exit with a clear
diagnostic message.

---

## Installation

```bash
git clone https://github.com/joshuahoffmann04/MARley.git
cd MARley

# 1. CUDA-enabled PyTorch first (the +cu121 suffix forces the GPU wheel)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Project + remaining dependencies
pip install -e .
```

Sanity check:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Configuration

### Environment variables

Copy the template and fill in only what you need:

```bash
cp .env.example .env
```

| Variable         | Used by                                     | Required?                 |
| ---------------- | ------------------------------------------- | ------------------------- |
| `OPENAI_API_KEY` | `--judge openai` (generation + E2E scoring) | Only for the OpenAI judge |

The default Ollama judge needs no configuration.

### Ollama setup

`OLLAMA_NUM_PARALLEL=2` is the sweet spot for an 8 B model on a 16 GB
GPU: two parallel slots saturate the GPU without VRAM contention. The
RAGAS judge batch size of 20 ( = 2 slots × 10 waves ) is tuned to match.

Ollama reads the variable **only at startup**, so a running instance
must be restarted after setting it.

#### Windows (PowerShell)

```powershell
# 1. Stop any existing Ollama processes
Stop-Process -Name 'ollama','ollama app' -Force -ErrorAction SilentlyContinue

# 2. Set the variable and start the server (keep this window open)
$env:OLLAMA_NUM_PARALLEL = "2"
ollama serve
```

To persist the setting across reboots, add `OLLAMA_NUM_PARALLEL=2` to
your user environment variables and restart Ollama once.

#### Linux / macOS

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

---

## Running the Chat Server

```powershell
# Terminal 1: Ollama with 2 parallel slots (see Configuration)
$env:OLLAMA_NUM_PARALLEL = "2"
ollama serve

# Terminal 2: MARley server
python -m src.marley.server --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Running the Evaluation

### Startup checklist

1. **Ollama** — running with `OLLAMA_NUM_PARALLEL=2`
   (see [Configuration](#ollama-setup)).
2. **Environment check** — one command validates GPU, eval inputs,
   chunk files, and Ollama reachability:
   ```bash
   python -m evaluation --check
   ```
3. **Run** — pick one of the commands below.

### Single-step commands

```bash
python -m evaluation --retrieval                   # Precision@k, Recall@k, MRR, F1@k, Jaccard@k
python -m evaluation --rrf-tuning                  # k_rrf sweep for Hybrid and Fusion
python -m evaluation --generation                  # Generation quality (RAGAS, Ollama judge)
python -m evaluation --abstention                  # Abstention precision/recall/F1
python -m evaluation --e2e                         # 33 configs × 100 questions, RAGAS-scored
```

### Full matrix

```bash
python -m evaluation --all --judge ollama --output-dir data/evaluation-ollama
python -m evaluation --all --judge openai --output-dir data/evaluation-openai
```

Each run is resumable: a crash mid-config is recovered by re-running the
same command (completed configs are skipped).

### Judge backends

The generator is always Ollama. The RAGAS judge that scores its answers
is swappable via `--judge`:

| Backend            | Model                   | Batch size | Requires                   |
| ------------------ | ----------------------- | ---------- | -------------------------- |
| `ollama` (default) | your local Ollama model | 20         | `OLLAMA_NUM_PARALLEL=2`    |
| `openai`           | `gpt-4o-mini`           | 50         | `OPENAI_API_KEY` in `.env` |

`--judge` takes effect in `--generation`, `--e2e`, and the corresponding
phases of `--all`. Abstention, retrieval, and RRF tuning measure
deterministic booleans and set operations — they ignore the flag.

In E2E, RAGAS scores only the samples where the system answered an
answerable question (Faithfulness, Answer Relevancy, Factual
Correctness). Abstentions and hallucinations on unanswerable questions
remain `NaN` on those fields and are excluded from the averages.

#### Decoupling the Ollama judge from the generator

By default, the Ollama judge uses the same model as the generator. For
better-calibrated scores — especially fewer NaN Faithfulness verdicts
on long or list-formatted answers — pair a small 8 B generator with a
larger judge via `--ollama-judge-model`:

```bash
ollama pull qwen2.5:14b
python -m evaluation --all --judge ollama \
    --ollama-model llama3.1:latest \
    --ollama-judge-model qwen2.5:14b \
    --output-dir data/evaluation-ollama
```

The generator and judge share the GPU. With 16 GB VRAM and
`OLLAMA_NUM_PARALLEL=2`, 8 B + 14 B at Q4 fits; larger judges (70 B)
require CPU offloading and slow the run substantially.

### Abstention-threshold selection

The Level-1 threshold sweep picks the threshold that maximises **F0.5**
(precision weighted 2× over recall). On the 25 %-unanswerable evaluation
dataset, F1-maximisation tended to choose very aggressive thresholds
(≥ 0.95) that traded away answers on answerable questions. F0.5 keeps
recall near its natural ceiling while preferring a slightly less
trigger-happy system — false abstentions dominate the user-experience
cost more than occasional hallucinations do.

### Quick iteration

For rapid iteration on the generation step:

| Flag                         | Effect                                                              |
| ---------------------------- | ------------------------------------------------------------------- |
| `--subset N`                 | Use only the first N questions per KB                               |
| `--distractor-levels 0,5,10` | Comma-separated distractor counts, overrides the default 0…10 sweep |
| `--kb-filter stpo`           | Restrict to a single KB, skips the combined-KB run                  |
| `--config-filter <substr>`   | Run only E2E configs whose name contains this substring             |

Example:

```bash
python -m evaluation --generation --subset 10 --distractor-levels 0,5,10 --judge openai
```

---

## Data Preparation

Run once after cloning — derived files land under `data/chunks/`:

```bash
# Extract sections and tables from the StPO PDF
python -c "from src.marley.extractor import extract, save; save(extract())"

# Chunk the extracted data (StPO + FAQ knowledge bases)
python -c "
from src.marley.chunker import chunk_stpo, save, chunk_faq, save_faq
save(chunk_stpo(), 'data/chunks/stpo-chunks.json')
save_faq(chunk_faq('data/knowledgebase/faq-stpo.json'), 'data/chunks/faq-stpo-chunks.json')
"
```

---

## Project Layout

```
src/marley/
├── models/          Shared data classes, ABCs, constants, scoring utilities
├── extractor/       PDF extraction (PyMuPDF + pdfplumber)
├── chunker/         PDF and FAQ chunking
├── retrieval/       BM25, Vector, Hybrid, Fusion, Merged retrievers
├── generator/       Ollama LLM backend
├── abstention/      Two-level abstention
└── server/          FastAPI chat server and pipeline orchestrator

evaluation/
├── retrieval/       Retrieval metrics + RRF tuning
├── generation/      Generation evaluation (RAGAS)
├── abstention/      Abstention evaluation (threshold sweep)
├── end_to_end/      End-to-end matrix evaluation (33 configs)
├── tests/           Evaluation unit tests
├── judge.py         RAGAS judge factory (Ollama / OpenAI)
├── validate.py      Data requirement validation
└── __main__.py      Unified evaluation CLI

tests/               Unit and integration tests (mirrored by component)

data/
├── raw/             Source PDFs
├── knowledgebase/   Extracted data + FAQ corpora
├── chunks/          Chunked JSONs + ChromaDB stores
└── evaluation/      Evaluation inputs and results

docs/
├── source/              Source code documentation
├── testing/             Test documentation
├── evaluation/          Evaluation documentation
├── evaluation-testing/  Evaluation test documentation
└── uml/                 UML class diagram
```

---

## Knowledge Bases

| KB         | Source                     | Chunks | Description                                             |
| ---------- | -------------------------- | ------ | ------------------------------------------------------- |
| `stpo`     | `msc-computer-science.pdf` | 153    | Study and examination regulations (text + tables)       |
| `faq-stpo` | `faq-stpo.json`            | 1039   | Synthetic FAQ derived from the StPO                     |
| `faq-ao`   | `faq-ao.json`              | 0      | Questions answered by the advisory office (placeholder) |

Evaluation datasets live in `data/evaluation/` (100 questions per KB,
plus a combined 100-question master set for E2E).

---

## Testing

**693 tests** across source code and evaluation — 688 passed, 5 skipped.
Overall coverage: **87.1 %** (3950 statements, 509 missed). The 5 skips
are integration tests that require a running Ollama server or the
optional `faq-ao` vector store.

```bash
# Full suite
python -m pytest tests/ evaluation/tests/

# Unit tests only
python -m pytest -m "not integration"

# Integration tests (requires data files + Ollama)
python -m pytest -m integration
```

<details>
<summary>Per-module breakdown (457 source + 236 evaluation)</summary>

### Source tests

| Component            | Tests | File                                 |
| -------------------- | ----- | ------------------------------------ |
| Extractor            | 81    | `tests/extractor/test_extractor.py`  |
| PDF Chunker          | 60    | `tests/chunker/test_pdf_chunker.py`  |
| FAQ Chunker          | 35    | `tests/chunker/test_faq_chunker.py`  |
| BM25 Retrieval       | 27    | `tests/retrieval/test_bm25.py`       |
| Vector Retrieval     | 25    | `tests/retrieval/test_vector.py`     |
| Hybrid Retrieval     | 26    | `tests/retrieval/test_hybrid.py`     |
| Fusion Retrieval     | 41    | `tests/retrieval/test_fusion.py`     |
| Merged Retrieval     | 14    | `tests/retrieval/test_merged.py`     |
| Generator            | 24    | `tests/generator/test_generator.py`  |
| Data Models          | 34    | `tests/models/test_models.py`        |
| Score Normalization  | 26    | `tests/models/test_scoring.py`       |
| Abstention Detection | 12    | `tests/abstention/test_detection.py` |
| Server API           | 17    | `tests/server/test_api.py`           |
| Server Models        | 15    | `tests/server/test_models.py`        |
| Server Pipeline      | 10    | `tests/server/test_pipeline.py`      |
| Server Service       | 10    | `tests/server/test_service.py`       |

### Evaluation tests

| Component           | Tests | File                                            |
| ------------------- | ----- | ----------------------------------------------- |
| Retrieval Metrics   | 33    | `evaluation/tests/retrieval/test_metrics.py`    |
| Retrieval Combined  | 25    | `evaluation/tests/retrieval/test_combined.py`   |
| Retrieval Evaluate  | 13    | `evaluation/tests/retrieval/test_evaluate.py`   |
| RRF Tuning          | 10    | `evaluation/tests/retrieval/test_rrf_tuning.py` |
| Generation Metrics  | 15    | `evaluation/tests/generation/test_metrics.py`   |
| Generation Evaluate | 19    | `evaluation/tests/generation/test_evaluate.py`  |
| Generation Combined | 15    | `evaluation/tests/generation/test_combined.py`  |
| Judge Factory       | 13    | `evaluation/tests/test_judge.py`                |
| Abstention Metrics  | 10    | `evaluation/tests/abstention/test_metrics.py`   |
| Abstention Evaluate | 12    | `evaluation/tests/abstention/test_evaluate.py`  |
| E2E Config          | 10    | `evaluation/tests/end_to_end/test_config.py`    |
| E2E Evaluate        | 22    | `evaluation/tests/end_to_end/test_evaluate.py`  |
| E2E Metrics         | 19    | `evaluation/tests/end_to_end/test_metrics.py`   |
| Utilities           | 20    | `evaluation/tests/test_utils.py`                |

</details>

Coverage gaps are deliberate: the FastAPI app routes and the evaluation
CLI dispatcher require a live Ollama server and are exercised by
integration tests rather than unit tests.

---

## Documentation

### Source

[Extractor](docs/source/extractor.md) ·
[Chunker](docs/source/chunker.md) ·
[Retrieval overview](docs/source/retrieval/overview.md)
([BM25](docs/source/retrieval/bm25.md),
[Vector](docs/source/retrieval/vector.md),
[Hybrid](docs/source/retrieval/hybrid.md),
[Fusion](docs/source/retrieval/fusion.md),
[Merged](docs/source/retrieval/merged.md)) ·
[Generator](docs/source/generator.md) ·
[Abstention](docs/source/abstention.md) ·
[Models](docs/source/models.md) ·
[Data](docs/source/data.md) ·
[Server](docs/source/server.md)

### Testing

[Overview](docs/testing/overview.md) ·
[Extractor](docs/testing/extractor.md) ·
[Chunker](docs/testing/chunker.md) ·
[Retrieval](docs/testing/retrieval.md) ·
[Generator](docs/testing/generator.md) ·
[Models](docs/testing/models.md) ·
[Abstention](docs/testing/abstention.md) ·
[Server](docs/testing/server.md)

### Evaluation

[Overview](docs/evaluation/overview.md) ·
[Retrieval](docs/evaluation/retrieval.md) ·
[Generation](docs/evaluation/generation.md) ·
[Abstention](docs/evaluation/abstention.md) ·
[End-to-End](docs/evaluation/end-to-end.md) ·
[Results](docs/evaluation/results/README.md)

### Evaluation tests

[Overview](docs/evaluation-testing/overview.md) ·
[Retrieval](docs/evaluation-testing/retrieval.md) ·
[Generation](docs/evaluation-testing/generation.md) ·
[Abstention](docs/evaluation-testing/abstention.md) ·
[End-to-End](docs/evaluation-testing/end-to-end.md) ·
[Utilities](docs/evaluation-testing/utils.md)

---

## License and Author

Licensed under the [MIT License](LICENSE).

Joshua Hoffmann — Bachelor thesis, Department of Mathematics and
Computer Science, Philipps-Universität Marburg, 2026.
