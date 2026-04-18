# Evaluation Results

> This directory holds the thesis-grade evaluation results once the
> final pipeline runs complete. The tree is intentionally empty ahead
> of those runs — prior artefacts have been cleared for Phase 11 so
> that only the final numbers appear here.

## Pipeline configuration for the final runs

| Component | Value |
|---|---|
| Generator | Ollama `llama3.1:latest` (8B, Q4_K_M) on CUDA |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` on CUDA |
| RAGAS version | 0.4.x |
| Judge (primary) | Ollama `llama3.1:latest` |
| Judge (alternative) | OpenAI `gpt-4o-mini` |
| Hardware | NVIDIA GeForce RTX 4070 Ti SUPER (16 GB) |

## Commands that produce these results

Two overnight runs, one per judge backend:

```bash
# 1. Local-only run (Ollama judge)
python -m evaluation --all --judge ollama \
    --output-dir data/evaluation-ollama

# 2. OpenAI-judge run
python -m evaluation --all --judge openai \
    --output-dir data/evaluation-openai
```

Each run writes:

| Artefact | Produced by |
|---|---|
| `retrieval-evaluation.json` | `--retrieval` |
| `rrf-tuning.json` | `--rrf-tuning` |
| `generation-evaluation.json` | `--generation` |
| `generation-evaluation-combined.json` | `--generation` (combined KB) |
| `abstention-evaluation.json` | `--abstention` |
| `e2e-results-{config-name}.json` | `--e2e` (one file per config) |

In the E2E artefacts, every non-abstained answerable answer carries
three RAGAS scores (Faithfulness, Answer Relevancy, Factual
Correctness) in addition to the abstention decision — see
[end-to-end.md § Answer Quality Scoring](../end-to-end.md#answer-quality-scoring).

## Files to appear here after the runs

| File | Source |
|---|---|
| `retrieval-results.md` | Summary of retrieval metrics |
| `generation-results.md` | RAGAS scores by KB × distractor level, both judges side by side |
| `abstention-results.md` | Level-1 sweep + final abstention metrics |
| `e2e-results.md` | Ranked comparison table across all 33 E2E configurations, with abstention and answer-quality columns |

Each summary is written from the JSON artefacts in
`data/evaluation-ollama/` and `data/evaluation-openai/`.
