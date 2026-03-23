# MARley

**MARburg Study Advising ChatBot** — A Retrieval-Augmented Generation (RAG) pipeline for answering questions about the M.Sc. Computer Science program at Philipps-Universität Marburg.

Bachelor thesis by Joshua Hoffmann, Department of Mathematics and Computer Science.

---

## Pipeline

```
PDF / FAQ Data
     │
     ├─ 1. Extractor     Extract sections and tables from the StPO PDF
     ├─ 2. Chunker        Split into retrieval-ready chunks (text, table, FAQ)
     ├─ 3. Retrieval      Find relevant chunks for a query (BM25 / Vector / Hybrid)
     ├─ 4. Generation     Generate an answer from the retrieved context
     ├─ 4.5 Abstention    Two-level controlled abstention (retrieval + LLM)
     └─ 5. Frontend       Chat interface for students
```

**Implemented:** Stages 1–5 + Evaluation harness (retrieval + generation + abstention + end-to-end).

---

## Project Structure

```
src/marley/
├── models/          Shared layer: data classes, ABCs, constants, scoring utilities
├── extractor/       PDF extraction (PyMuPDF + pdfplumber)
├── chunker/         PDF chunking (sentence-aligned sliding window) + FAQ chunking
├── retrieval/       BM25, Vector, Hybrid (RRF), and Fusion retrievers
├── generator/       Ollama LLM backend for answer generation
├── abstention/      LLM-level abstention detection
└── server/          Pipeline orchestrator + FastAPI server (chat UI, debug UI, API)

evaluation/
├── retrieval/       Retrieval evaluation (Precision@k, Recall@k, MRR) + RRF tuning
├── generation/      Generation evaluation (distractor robustness)
├── manual/          Manual evaluation framework (human correctness assessment)
├── abstention/      Abstention evaluation (precision, recall, F1)
├── end_to_end/      End-to-end pipeline evaluation (33 configs × 100 questions)
├── validate.py      Data requirement validation
└── __main__.py      Unified evaluation CLI (python -m evaluation)

tests/               Unit and integration tests (mirrored by component)
docs/                Component documentation (mirrored by component)

data/
├── raw/             Source PDFs
├── knowledgebase/   Extracted data + FAQ knowledge bases
├── chunks/          Chunked output (retrieval-ready JSON)
└── testing/         Evaluation datasets (100 questions × 3 knowledge bases)
```

---

## Knowledge Bases

| Knowledge Base | Source | Chunks | Description |
|---|---|---|---|
| StPO | `msc-computer-science.pdf` | 153 | Study and examination regulations (text + tables) |
| FAQ-StPO | `faq-stpo.json` | 1039 | Synthetic FAQ derived from the StPO |
| FAQ-AO | `faq-ao.json` | 0 | Student questions answered by the advisory office (placeholder, pending data) |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest

# Run unit tests only (fast, no data files needed)
python -m pytest -m "not integration"

# Run extraction + chunking
python -c "
from src.marley.extractor import extract, save
result = extract('data/raw/msc-computer-science.pdf')
save(result, 'data/knowledgebase/stpo-extracted.json')
"

# Run BM25 retrieval
python -c "
from src.marley.retrieval import BM25Retriever, load_chunks
chunks = load_chunks('data/chunks/stpo-chunks.json')
retriever = BM25Retriever()
retriever.index(chunks)
for r in retriever.retrieve('master thesis credits', k=5):
    print(f'{r.chunk_id}: {r.score:.2f}')
"

# Start the server (requires Ollama running)
python -m src.marley.server --port 8000

# Run evaluation (unified CLI)
python -m evaluation --check              # Validate data requirements
python -m evaluation --retrieval          # Retrieval evaluation only
python -m evaluation --rrf-tuning         # Sweep k_rrf for Hybrid and Fusion
python -m evaluation --all                # Run all evaluation steps
```

---

## Tests

```bash
python -m pytest                         # Run all 634 tests
python -m pytest -m "not integration"    # Unit tests only (505, fast, no data needed)
python -m pytest -m integration          # Integration tests only (129, requires data files)
```

| Component | Tests | Test file |
|---|---|---|
| Data Models | 20 | `tests/models/test_models.py` |
| Score Normalization + Filtering | 20 | `tests/models/test_scoring.py` |
| Extractor | 81 | `tests/extractor/test_extractor.py` |
| PDF Chunker | 60 | `tests/chunker/test_pdf_chunker.py` |
| FAQ Chunker | 35 | `tests/chunker/test_faq_chunker.py` |
| BM25 Retrieval | 23 | `tests/retrieval/test_bm25.py` |
| Vector Retrieval | 23 | `tests/retrieval/test_vector.py` |
| Hybrid Retrieval | 23 | `tests/retrieval/test_hybrid.py` |
| RRF Fusion + FusionRetriever | 28 | `tests/retrieval/test_fusion.py` |
| Generator | 23 | `tests/generator/test_generator.py` |
| Abstention Detection | 12 | `tests/abstention/test_detection.py` |
| Abstention Pipeline | 10 | `tests/server/test_pipeline.py` |
| Server (Models + Service + API + Pipeline) | 52 | `tests/server/test_models.py`, `tests/server/test_service.py`, `tests/server/test_api.py`, `tests/server/test_pipeline.py` |
| Retrieval Evaluation | 34 | `evaluation/tests/retrieval/test_metrics.py`, `evaluation/tests/retrieval/test_evaluate.py` |
| Combined Retrieval Evaluation | 23 | `evaluation/tests/retrieval/test_combined.py` |
| RRF k-Parameter Tuning | 10 | `evaluation/tests/retrieval/test_rrf_tuning.py` |
| Generation Evaluation | 22 | `evaluation/tests/generation/test_metrics.py`, `evaluation/tests/generation/test_evaluate.py` |
| Combined Generation Evaluation | 14 | `evaluation/tests/generation/test_combined.py` |
| Manual Evaluation | 46 | `evaluation/tests/manual/test_models.py`, `evaluation/tests/manual/test_prepare.py`, `evaluation/tests/manual/test_metrics.py` |
| Abstention Evaluation | 22 | `evaluation/tests/abstention/test_metrics.py`, `evaluation/tests/abstention/test_evaluate.py` |
| E2E Evaluation | 47 | `evaluation/tests/end_to_end/test_*.py` |
| Evaluation Utilities | 16 | `evaluation/tests/test_utils.py` |
| **Total** | **634** | |

Integration tests (`@pytest.mark.integration`) require data files or external services and are skipped automatically when using `-m "not integration"`.

---

## Documentation

| Document | Path |
|---|---|
| Data Models | `docs/models/models.md` |
| PDF Extractor | `docs/extractor/extractor.md` |
| PDF Chunker | `docs/chunker/pdf_chunker.md` |
| FAQ Chunker | `docs/chunker/faq_chunker.md` |
| BM25 Retrieval | `docs/retrieval/bm25.md` |
| Vector Retrieval | `docs/retrieval/vector.md` |
| Hybrid Retrieval | `docs/retrieval/hybrid.md` |
| Fusion Retrieval | `docs/retrieval/fusion.md` |
| Generator | `docs/generator/generator.md` |
| Evaluation Overview | `docs/evaluation/overview.md` |
| Retrieval Evaluation | `docs/evaluation/retrieval.md` |
| Combined Retrieval Evaluation | `docs/evaluation/combined-retrieval.md` |
| Generation Evaluation | `docs/evaluation/generation.md` |
| Combined Generation Evaluation | `docs/evaluation/combined-generation.md` |
| Manual Evaluation | `docs/evaluation/manual-evaluation.md` |
| Abstention Pipeline | `docs/abstention/abstention.md` |
| Abstention Evaluation | `docs/evaluation/abstention.md` |
| End-to-End Evaluation | `docs/evaluation/end-to-end.md` |
| Server | `docs/server/server.md` |
| Data Structures | `docs/data/data-structures.md` |
| FAQ Coverage Plan | `docs/data/faq-stpo-coverage.md` |

Test documentation mirrors the component structure under `docs/testing/`.

---

## Dependencies

- Python 3.12+
- PyMuPDF, pdfplumber (PDF extraction)
- syntok, tiktoken (text processing)
- rank-bm25 (sparse retrieval)
- sentence-transformers, chromadb (dense retrieval)
- ollama (local LLM generation)

See `requirements.txt` for the full list.
