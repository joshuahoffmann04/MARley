# Evaluation Results

> Phase 8 evaluation results for the MARley RAG pipeline.
> Model: llama3.1:latest (8B, Q4_K_M) on CPU | RAGAS 0.4.3

## Completed Evaluations

| File                                           | Content                                                           | Status   |
| ---------------------------------------------- | ----------------------------------------------------------------- | -------- |
| [retrieval-results.md](retrieval-results.md)   | Retrieval metrics (12 configs: 3 retrievers x 2 KBs + 6 multi-KB) | Complete |
| [generation-results.md](generation-results.md) | RAGAS scores (2 KBs x 11 distractor levels = 1650 samples)        | Complete |

## Pending Evaluations

| Evaluation | CLI Command                         |
| ---------- | ----------------------------------- |
| Abstention | `python -m evaluation --abstention` |
| End-to-End | `python -m evaluation --e2e`        |

## Data Files

All JSON results are stored in `data/evaluation/`:

| File                         | Size   | Content                                            |
| ---------------------------- | ------ | -------------------------------------------------- |
| `retrieval-evaluation.json`  | 6 KB   | 12 retriever configs with P@5, R@5, MRR, F1@5, J@5 |
| `rrf-tuning.json`            | 17 KB  | k_rrf sweep (11 values x 4 configs)                |
| `generation-evaluation.json` | 1.8 MB | 1650 samples with per-sample RAGAS scores          |
