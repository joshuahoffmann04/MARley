# End-to-End Evaluation

> Evaluates the full pipeline (retrieval → abstention → generation) across 33 configurations.

## Configuration Matrix

The E2E evaluation tests every combination of retriever type, knowledge base selection, and retrieval strategy.

### 33 Configurations

| Strategy | KBs | Retrievers | Count |
|---|---|---|---|
| **Single** | 3 (stpo, faq-stpo, faq-ao) | 3 (BM25, Vector, Hybrid) | 9 |
| **Merged Pool** | 4 combinations | 3 (BM25, Vector, Hybrid) | 12 |
| **Fusion** | 4 combinations | 3 (BM25, Vector, Hybrid) | 12 |
| | | **Total** | **33** |

### Knowledge Base Combinations (Merged Pool + Fusion)

1. stpo + faq-stpo
2. stpo + faq-ao
3. faq-stpo + faq-ao
4. stpo + faq-stpo + faq-ao (all)

### Retriever Types

| Type | Normalization Strategy |
|---|---|
| `bm25` | `bm25` |
| `vector` | `vector` |
| `hybrid` | `rrf` |

For **fusion** strategy, normalization is always `rrf` regardless of retriever type.

### E2EConfig Dataclass



Implementation: `evaluation/end_to_end/config.py` (`generate_all_configs()`)

### Naming Convention

Config names follow the pattern: `{strategy}-{kb_label}-{retriever_type}`

Examples:
- `single-stpo-bm25` — BM25 on StPO alone
- `merged-all-vector` — Vector on all KBs merged
- `fusion-stpo+faq-stpo-hybrid` — Hybrid retriever per KB, fused via RRF

## Pipeline Per Configuration

For each of the 33 configurations, the E2E evaluation runs:

### Step 1: Build Retriever

Based on the strategy:
- **Single / Merged Pool**: Create one retriever, index on (merged) chunks
- **Fusion**: Create one retriever per KB, wrap in `FusionRetriever`

Implementation: `evaluation/end_to_end/run_all.py` (`build_retriever()`)

### Step 2: Sweep Threshold (No LLM)

Optimize the Level 1 abstention threshold without LLM calls:



Default sweep: 0.0 to 1.0 in 0.05 steps. Selects threshold maximizing F1.

Implementation: `evaluation/end_to_end/evaluate.py` (`sweep_threshold()`)

### Step 3: Run Full Pipeline

For each question in the evaluation dataset:

1. **Retrieve**: Get top-k chunks from the (already-indexed) retriever
2. **Normalize**: Apply score normalization for the config's strategy
3. **Level 1 Check**: Filter by confidence threshold
4. **Generate** (if passed): Generate answer using Ollama
5. **Level 2 Check**: Detect `ABSTENTION:` prefix in the generated answer
6. **Record**: Store `E2EResult` with answer, abstention status, confidence

Implementation: `evaluation/end_to_end/evaluate.py` (`run_e2e_config()`)

### Step 4: Compute Metrics and Save

- Compute abstention metrics via `compute_abstention_metrics()`
- Save report JSON with config, threshold, sweep results, and per-question results

### E2EResult Dataclass



## Aggregation and Comparison

### E2EConfigMetrics

Automatic metrics per configuration (no human judgement required):



Implementation: `evaluation/end_to_end/metrics.py` (`compute_e2e_config_metrics()`)

### Comparison Table

`build_comparison_table()` produces a sortable comparison across all configs:



Returns rows sorted by `abstention_f1` descending, with columns:
`config_name`, `num_total`, `num_abstained`, `abstention_rate`,
`abstention_precision`, `abstention_recall`, `abstention_f1`,
`avg_confidence`, `level1_abstentions`, `level2_abstentions`

### Per-Category Breakdown

Each config's metrics include a per-category abstention breakdown:



## Resume Support

The runner supports resuming interrupted evaluations:
- Each config produces a separate output file: `e2e-results-{config.name}.json`
- On restart, configs with existing output files are skipped
- Progress is logged: `[N/33] START config-name` or `[N/33] SKIP config-name (output exists)`

## CLI Usage



The E2E module also has its own standalone CLI:

usage: run_all.py [-h] [--output-dir OUTPUT_DIR] [--ollama-url OLLAMA_URL]
                  [--ollama-model OLLAMA_MODEL]
                  [--config-filter CONFIG_FILTER]

Run all MARley end-to-end evaluation configurations.

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory for output files (default: data/evaluation)
  --ollama-url OLLAMA_URL
                        Ollama server URL (default: http://localhost:11434)
  --ollama-model OLLAMA_MODEL
                        Ollama model name (default: llama3.1:latest)
  --config-filter CONFIG_FILTER
                        Only run configs whose name contains this substring

## Output Files

One JSON file per configuration:



### Report Format


