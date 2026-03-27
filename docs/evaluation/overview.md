# Evaluation Overview

**Module:** `evaluation/`
**Entry point:** `python -m evaluation [OPTIONS]`

The evaluation framework measures the quality of each pipeline stage independently and as a whole. All evaluations use the same annotated dataset of 100 questions across three categories: direct (40), multi-source (35), and unanswerable (25).

---

## Evaluation Types

| # | Evaluation | Module | Document | Status |
|---|---|---|---|---|
| 1 | Single-KB Retrieval | `evaluation/retrieval/evaluate.py` | [retrieval.md](retrieval.md) | Complete |
| 1a | RRF k_rrf Tuning | `evaluation/retrieval/rrf_tuning.py` | [rrf-tuning.md](rrf-tuning.md) | Complete |
| 2 | Combined-KB Retrieval | `evaluation/retrieval/combined.py` | [combined-retrieval.md](combined-retrieval.md) | Complete |
| 3 | Generation | `evaluation/generation/` | [generation.md](generation.md) | Complete |
| 3a | Manual Evaluation | `evaluation/manual/` | [manual-evaluation.md](manual-evaluation.md) | Complete |
| 4 | Combined-KB Generation | `evaluation/generation/combined.py` | [combined-generation.md](combined-generation.md) | Complete |
| 5 | Abstention | `evaluation/abstention/` | [abstention.md](abstention.md) | Complete |
| 6 | End-to-End | `evaluation/end_to_end/` | [end-to-end.md](end-to-end.md) | Complete |

---

## Evaluation Strategy

### Proposal Goals

The evaluation framework is structured around the three proposal goals:

- **G1 (Knowledge Preparation):** Measured indirectly through retrieval quality — better chunking leads to better retrieval metrics.
- **G2 (Retrieval Strategy Comparison):** Single-KB evaluation (9 configurations) and combined-KB evaluation (24 configurations) compare BM25, Vector, and Hybrid retrieval across all knowledge bases and combination strategies.
- **G3 (Generation + Abstention):** Generation evaluation measures answer quality with varying context quality. Abstention evaluation measures the system's ability to refuse unanswerable questions.

### Knowledge Bases

| Knowledge Base | Source | Chunks | Description |
|---|---|---|---|
| StPO | `msc-computer-science.pdf` | 153 | Study and examination regulations (text + tables) |
| FAQ-StPO | `faq-stpo.json` | 1039 | Synthetic FAQ derived from the StPO |
| FAQ-AO | `faq-ao.json` | 0 | Student questions answered by the advisory office (placeholder, no chunks yet) |

### Shared Metrics

**Retrieval metrics** (used by evaluations 1, 2):
- **Precision@k** — proportion of top-k results that are relevant
- **Recall@k** — proportion of all relevant chunks found in top-k
- **F1@k** — harmonic mean of Precision@k and Recall@k
- **MRR** — reciprocal rank of the first relevant result
- **MAP** — mean average precision across all relevant hit positions
- **Jaccard@k** — set overlap between retrieved and relevant chunks

All retrieval metrics are macro-averaged over evaluated queries.

### Evaluation Dataset

All evaluations draw from the same master dataset of 100 questions (`data/testing/evaluation.json`), with per-KB annotations in:
- `data/testing/evaluation-stpo.json` (75 evaluable questions)
- `data/testing/evaluation-faq-stpo.json` (75 evaluable questions)
- `data/testing/evaluation-faq-ao.json` (21 evaluable questions)

---

## CLI Entry Point

The unified CLI (`evaluation/__main__.py`) provides a single entry point for all evaluation steps.

### Usage

```bash
python -m evaluation [OPTIONS]
```

### Step Selection Flags

| Flag | Description |
|---|---|
| `--check` | Validate data requirements only (no evaluation). |
| `--retrieval` | Run retrieval evaluation (single-KB + combined). |
| `--rrf-tuning` | Sweep `k_rrf` for Hybrid and Fusion retrievers. |
| `--generation` | Run generation evaluation. |
| `--abstention` | Run abstention evaluation (Level 1 sweep + full). |
| `--e2e` | Run end-to-end evaluation. |
| `--all` | Run all steps in order. |

### Common Options

| Option | Default | Description |
|---|---|---|
| `--output-dir` | `data/testing` | Output directory for evaluation results. |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL. |
| `--ollama-model` | `llama3.1:latest` | LLM model name for generation/abstention/E2E. |
| `--config-filter` | `None` | Only run E2E configs matching this substring. |

### Execution Order

When `--all` is used, steps execute in this order: `retrieval` -> `rrf-tuning` -> `generation` -> `abstention` -> `e2e`. Each step validates data requirements before running.

---

## Shared Utilities

### `evaluation/utils.py`

Common functions used across all evaluation modules.

| Function / Class | Description |
|---|---|
| `load_json(path)` | Load a JSON file and return its contents as a dict. |
| `load_evaluation(eval_path)` | Load an annotated evaluation JSON file; returns the list of question dicts. |
| `merge_chunks(*chunk_paths)` | Load and concatenate chunks from multiple files; raises `ValueError` on duplicate chunk IDs. |
| `merge_evaluation_data(eval_paths)` | Merge evaluation datasets across KBs; unions `relevant_chunks` per question. |
| `compute_abstention_metrics(results, threshold)` | Compute abstention precision, recall, F1, false abstention rate, and coverage. |
| `AbstentionMetrics` | Dataclass holding abstention evaluation results (precision, recall, F1, false_abstention_rate, coverage, counts, threshold). |

### `evaluation/validate.py`

Pre-flight validation that checks all required files and services before running evaluation steps.

| Symbol | Description |
|---|---|
| `EVAL_PATHS` | Dict mapping KB names to evaluation file paths (`data/testing/evaluation-*.json`). |
| `OLLAMA_STEPS` | Set of step names requiring Ollama (`{"generation", "abstention", "e2e"}`). |
| `validate_data_requirements(steps, output_dir, ollama_url)` | Returns a list of error messages (empty = all OK). Checks chunk files, evaluation files, and Ollama connectivity. |

---

## Key Findings Summary

### Single-KB Retrieval

Vector retrieval outperforms BM25 across all knowledge bases. Hybrid (RRF) achieves the best recall but lower precision than pure Vector. See [retrieval.md](retrieval.md) for full results.

### Combined-KB Retrieval

Combining knowledge bases generally maintains or improves retrieval quality compared to single-KB baselines. The fusion strategy (per-KB retrievers + RRF) slightly outperforms the merged pool strategy for BM25 and Hybrid, while both strategies perform similarly for Vector. See [combined-retrieval.md](combined-retrieval.md) for full results.

---

## Test Coverage

| Test File | Tests | Evaluation Covered |
|---|---|---|
| `evaluation/tests/retrieval/test_metrics.py` | 41 | Retrieval metrics |
| `evaluation/tests/retrieval/test_evaluate.py` | 13 | Single-KB evaluation runner |
| `evaluation/tests/retrieval/test_combined.py` | 25 | Combined-KB evaluation runner |
| `evaluation/tests/retrieval/test_rrf_tuning.py` | 10 | RRF k-parameter sweep |
| `evaluation/tests/generation/test_metrics.py` | 5 | Generation metrics |
| `evaluation/tests/generation/test_evaluate.py` | 17 | Generation evaluation runner |
| `evaluation/tests/generation/test_combined.py` | 14 | Combined-KB generation runner |
| `evaluation/tests/manual/test_models.py` | 23 | Manual evaluation data model |
| `evaluation/tests/manual/test_prepare.py` | 10 | Manual evaluation item preparation |
| `evaluation/tests/manual/test_metrics.py` | 13 | Manual evaluation metrics |
| `evaluation/tests/abstention/test_metrics.py` | 10 | Abstention metrics |
| `evaluation/tests/abstention/test_evaluate.py` | 12 | Abstention evaluation runner |
| `evaluation/tests/end_to_end/test_config.py` | 10 | E2E configuration generation |
| `evaluation/tests/end_to_end/test_evaluate.py` | 17 | E2E evaluation runner |
| `evaluation/tests/end_to_end/test_prepare.py` | 8 | E2E item preparation |
| `evaluation/tests/end_to_end/test_metrics.py` | 12 | E2E metrics aggregation |
| `evaluation/tests/test_utils.py` | 16 | Shared utilities |
| **Total** | **256** | |
