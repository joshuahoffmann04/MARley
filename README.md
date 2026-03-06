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
     └─ 5. Frontend       Chat interface for students
```

**Implemented:** Stages 1–3 + Evaluation harness.

---

## Project Structure

```
src/marley/
├── models/          Shared data classes (ExtractionResult, QualityFlag, ...)
├── extractor/       PDF extraction (PyMuPDF + pdfplumber)
├── chunker/         PDF chunking (sentence-aligned) + FAQ chunking
└── retrieval/       Abstract Retriever interface + BM25 + Vector + Hybrid (RRF)

evaluation/
└── retrieval/       Retrieval evaluation (Precision@k, Recall@k, MRR)

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
| StPO | `msc-computer-science.pdf` | 142 | Study and examination regulations (text + tables) |
| FAQ-StPO | `faq-stpo.json` | 999 | Synthetic FAQ derived from the StPO |
| FAQ-AO | `faq-ao.json` | 50 | Student questions answered by the advisory office |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ evaluation/tests/ -v

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

# Run evaluation
python -c "
from src.marley.retrieval import BM25Retriever, load_chunks
from evaluation.retrieval.evaluate import run_and_report
chunks = load_chunks('data/chunks/stpo-chunks.json')
retriever = BM25Retriever()
retriever.index(chunks)
report = run_and_report(retriever, 'data/testing/evaluation-stpo.json', k=5)
print(report['metrics'])
"
```

---

## Tests

| Component | Tests | Test file |
|---|---|---|
| Extractor | 66 | `tests/extractor/test_extractor.py` |
| PDF Chunker | 50 | `tests/chunker/test_pdf_chunker.py` |
| FAQ Chunker | 36 | `tests/chunker/test_faq_chunker.py` |
| BM25 Retrieval | 23 | `tests/retrieval/test_bm25.py` |
| Vector Retrieval | 23 | `tests/retrieval/test_vector.py` |
| Hybrid Retrieval | 23 | `tests/retrieval/test_hybrid.py` |
| Evaluation | 34 | `evaluation/tests/retrieval/test_metrics.py`, `evaluation/tests/retrieval/test_evaluate.py` |
| **Total** | **255** | |

Integration tests that require data files are skipped automatically in CI.

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
| Retrieval Evaluation | `docs/evaluation/retrieval.md` |
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

See `requirements.txt` for the full list.
