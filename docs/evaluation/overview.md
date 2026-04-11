# Evaluation Overview

> Documentation of the MARley evaluation pipeline.
> Covers retrieval, generation, abstention, and end-to-end evaluation.

## Evaluation Strategy

The MARley evaluation framework systematically measures pipeline quality across four dimensions:

| Dimension | What Is Evaluated | Metrics |
|---|---|---|
| **Retrieval** | Chunk retrieval accuracy | Precision@k, Recall@k, MRR, F1@k, Jaccard@k |
| **Generation** | Answer quality (via RAGAS) | Faithfulness, Answer Relevancy, Factual Correctness |
| **Abstention** | Abstention decision quality | Precision, Recall, F1, False Abstention Rate, Coverage |
| **End-to-End** | Full pipeline (33 configs) | Abstention metrics + per-category breakdown |

Each dimension can be evaluated independently or as part of a full evaluation run.

## Metrics Landscape

### Retrieval (5 metrics)

All metrics are computed at cutoff `k` (default: 5) and macro-averaged over all answerable queries.

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision@k** | \|relevant ∩ retrieved[:k]\| / k | Fraction of retrieved chunks that are relevant |
| **Recall@k** | \|relevant ∩ retrieved[:k]\| / \|relevant\| | Fraction of relevant chunks that were retrieved |
| **MRR** | 1 / rank of first relevant chunk | How quickly the first relevant chunk appears |
| **F1@k** | 2 · P@k · R@k / (P@k + R@k) | Harmonic mean of precision and recall |
| **Jaccard@k** | \|relevant ∩ retrieved[:k]\| / \|relevant ∪ retrieved[:k]\| | Set similarity between relevant and retrieved |

Implementation: `evaluation/retrieval/metrics.py`

### Generation (3 RAGAS metrics)

Quality is measured via [RAGAS](https://docs.ragas.io/) using a local Ollama LLM as the evaluator.

| Metric | What It Measures | Scale |
|---|---|---|
| **Faithfulness** | Answer only uses information from the provided context | 0-1 |
| **Answer Relevancy** | Answer directly addresses the user's question | 0-1 |
| **Factual Correctness** | Answer matches the reference answer | 0-1 |

Implementation: `evaluation/generation/evaluate.py` (RAGAS scoring in `_score_with_ragas()`)

### Abstention (5 metrics)

Evaluates the two-level abstention mechanism (Level 1: retrieval confidence, Level 2: LLM detection).

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision** | correct_abstention / all_abstentions | When the system abstains, is it right? |
| **Recall** | correct_abstention / all_unanswerable | Does the system catch unanswerable questions? |
| **F1** | 2 · Precision · Recall / (Precision + Recall) | Harmonic mean of precision and recall |
| **False Abstention Rate** | incorrect_abstention / all_answerable | How often does it wrongly refuse to answer? |
| **Coverage** | (answered + missing_abstention) / total | Proportion of questions receiving an answer |

Implementation: `evaluation/utils.py` (`AbstentionMetrics`, `compute_abstention_metrics()`)

## CLI Usage

All evaluation is driven through a unified CLI:

```bash
python -m evaluation --help
```

### Available Commands

```bash
# Validate data requirements (chunks, eval files, Ollama)
python -m evaluation --check

# Individual evaluation steps
python -m evaluation --retrieval       # Retrieval metrics (P@k, R@k, MRR, F1@k, Jaccard@k)
python -m evaluation --rrf-tuning      # k_rrf parameter sweep for Hybrid and Fusion
python -m evaluation --generation      # Generation quality via RAGAS
python -m evaluation --abstention      # Abstention metrics with threshold sweep
python -m evaluation --e2e             # End-to-end (33 configs x 100 questions)

# Run all steps in order
python -m evaluation --all
```

### Common Options

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `data/evaluation` | Directory for evaluation output files |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--ollama-model` | `llama3.1:latest` | Ollama model for generation and RAGAS |
| `--config-filter` | `None` | Only run E2E configs matching this substring |

### Execution Order

When using `--all`, steps run in this order:

```
retrieval -> rrf-tuning -> generation -> abstention -> e2e
```

Each step validates data requirements before running. Steps that need an LLM (`generation`, `abstention`, `e2e`) require a running Ollama server.

## Data Prerequisites

### Required Files

| File | Purpose |
|---|---|
| `data/chunks/stpo-chunks.json` | StPO chunk corpus |
| `data/chunks/faq-stpo-chunks.json` | FAQ-StPO chunk corpus |
| `data/chunks/faq-ao-chunks.json` | FAQ-AO chunk corpus |
| `data/evaluation/evaluation-stpo.json` | StPO evaluation dataset |
| `data/evaluation/evaluation-faq-stpo.json` | FAQ-StPO evaluation dataset |
| `data/evaluation/evaluation-faq-ao.json` | FAQ-AO evaluation dataset |

### Evaluation Dataset Format

Each evaluation JSON file contains:

```json
{
  "questions": [
    {
      "id": "q-001",
      "question": "What are the admission requirements?",
      "reference_answer": "The program requires...",
      "relevant_chunks": ["stpo-sec-3-chunk-1", "stpo-sec-3-chunk-2"],
      "category": "admission",
      "expected_abstention": false
    }
  ]
}
```

### Ollama Requirement

Steps that involve generation (`--generation`, `--abstention`, `--e2e`) require a running Ollama server. The `--check` command validates Ollama availability.

## Output Files

All evaluation results are saved as JSON in the output directory:

| File | Produced By |
|---|---|
| `retrieval-evaluation.json` | `--retrieval` |
| `rrf-tuning.json` | `--rrf-tuning` |
| `generation-evaluation.json` | `--generation` |
| `generation-evaluation-combined.json` | `--generation` (combined KB) |
| `abstention-evaluation.json` | `--abstention` |
| `e2e-results-{config-name}.json` | `--e2e` (one file per config) |

## Module Structure

```
evaluation/
+-- __init__.py              # Package init, exports validate_data_requirements
+-- __main__.py              # Unified CLI entry point
+-- validate.py              # Data requirement validation (EVAL_PATHS, CHUNK_PATHS)
+-- utils.py                 # Shared utilities (load_json, merge_chunks, AbstentionMetrics)
+-- retrieval/
|   +-- metrics.py           # RetrievalMetrics dataclass + 5 metric functions
|   +-- evaluate.py          # Single-KB retrieval evaluation runner
|   +-- combined.py          # Merged pool + Fusion evaluation strategies
|   +-- rrf_tuning.py        # k_rrf parameter sweep (Hybrid + Fusion)
+-- generation/
|   +-- __init__.py          # Public API re-exports
|   +-- metrics.py           # GenerationEvalResult + GenerationMetrics dataclasses
|   +-- evaluate.py          # Single-KB generation evaluation + RAGAS scoring
|   +-- combined.py          # Combined-KB generation evaluation
+-- abstention/
|   +-- evaluate.py          # Level 1 sweep + full two-level evaluation
|   +-- metrics.py           # Re-exports AbstentionMetrics from utils.py
+-- end_to_end/
|   +-- __init__.py          # Package init
|   +-- config.py            # E2EConfig dataclass + 33-config generator
|   +-- evaluate.py          # E2EResult + sweep_threshold + run_e2e_config
|   +-- metrics.py           # E2EConfigMetrics + comparison table builder
|   +-- run_all.py           # Main runner with resume support
+-- tests/                   # Evaluation unit tests (see evaluation-testing docs)
```

## Related Documentation

- [Retrieval Evaluation](retrieval.md) -- Detailed retrieval metrics and strategies
- [Generation Evaluation](generation.md) -- RAGAS integration and distractor testing
- [Abstention Evaluation](abstention.md) -- Two-level abstention evaluation
- [End-to-End Evaluation](end-to-end.md) -- 33-config matrix evaluation
- [Results](results/) -- Evaluation results (populated in Phase 8)
