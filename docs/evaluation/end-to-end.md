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



Default sweep: 0.0 to 1.0 in 0.05 steps. Selects threshold maximising
**F0.5** (precision weighted 2× over recall). On the 25 %-unanswerable
evaluation dataset, F1-maximisation tended to choose very aggressive
thresholds (≥ 0.95) that traded away answers on answerable questions;
F0.5 keeps recall near its natural ceiling while preferring a slightly
less trigger-happy abstention.

For :class:`FusionRetriever` configs, the sweep uses a **Fusion-aware
confidence** computed from the sub-retrievers' top-1 normalised scores
(see [fusion.md](../source/retrieval/fusion.md#fusion-aware-confidence))
rather than the degenerate fused RRF score. The abstention decision is
then all-or-nothing per query, gated by that confidence.

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

### Step 4: Score Answers with RAGAS

After all questions are generated, the runner batches every
non-abstained answerable answer (Option A — see "Answer Quality
Scoring" below) and hands them to `_score_e2e_answers()`. This function
shares its RAGAS stack with the generation eval, so Ollama and OpenAI
judges behave identically in both pipelines. Scores are written back
into the matching `E2EResult` slots; samples outside the scoring scope
keep their `NaN` defaults.

### Step 5: Compute Metrics and Save

- Compute abstention metrics via `compute_abstention_metrics()`.
- Compute generation metrics (`num_scored`, `avg_faithfulness`,
  `avg_answer_relevance`, `avg_correctness`) via
  `_aggregate_generation_metrics()` — NaN-excluded averages over the
  scored subset.
- Save report JSON with config, threshold, sweep results, abstention
  metrics, generation metrics, `judge_batch_size`, and per-question
  results (each carrying its three RAGAS fields, NaN when unscored).

### E2EResult Dataclass



## Answer Quality Scoring

Every E2E configuration reports **two metric families** side by side:
abstention decisions (deterministic) and answer quality (RAGAS).
Phase-11 "Option A" defines the scoring scope so each score has a
meaningful reference:

| `expected_abstention` | `abstained` | Scored? | Why |
|---|---|---|---|
| False | False | **yes** | Answerable, answered — has both a generated answer and a reference |
| False | True | no | Answerable but refused — no text to judge |
| True | True | no | Correct abstention — no text to judge |
| True | False | no | Hallucination — no reference, Factual Correctness undefined |

Only cell 1 receives RAGAS scores. All others keep `NaN` on the three
fields (`faithfulness`, `answer_relevance`, `correctness`) and are
excluded from the averages. The RAGAS judge is selected with
`--judge ollama` (default, local) or `--judge openai`
(`gpt-4o-mini`) — identical to the generation eval.

Implementation: `evaluation/end_to_end/evaluate.py::_score_e2e_answers`.

## Aggregation and Comparison

### E2EConfigMetrics

Automatic metrics per configuration (no human judgement required).
Abstention fields come from the orchestration decisions; generation
fields come from the Option-A scored subset:

| Field | Source |
|---|---|
| `abstention_precision`, `abstention_recall`, `abstention_f1` | Confusion matrix over `expected_abstention` vs. `abstained` |
| `level1_abstentions`, `level2_abstentions` | Per-level counts |
| `abstention_by_category` | Same computation, grouped by `category` |
| `num_scored` | Samples that received RAGAS scores |
| `avg_faithfulness`, `avg_answer_relevance`, `avg_correctness` | Means over non-NaN RAGAS values |

Implementation: `evaluation/end_to_end/metrics.py` (`compute_e2e_config_metrics()`)

### Comparison Table

`build_comparison_table()` produces a sortable comparison across all
configs. Rows are sorted by `abstention_f1` descending (orchestration
ranking); callers can re-sort by `avg_correctness` for an answer-quality
ranking. Each row contains every abstention column plus the four
generation columns: `num_scored`, `avg_faithfulness`,
`avg_answer_relevance`, `avg_correctness`.

### Per-Category Breakdown

Each config's metrics include a per-category abstention breakdown:



## Resume Support

The runner supports resuming interrupted evaluations:
- Each config produces a separate output file: `e2e-results-{config.name}.json`
- On restart, configs with existing output files are skipped
- Progress is logged: `[N/33] START config-name` or `[N/33] SKIP config-name (output exists)`

## CLI Usage



Prefer the unified CLI:

```bash
python -m evaluation --e2e                       # Ollama judge (default)
python -m evaluation --e2e --judge openai        # OpenAI gpt-4o-mini judge
python -m evaluation --e2e --config-filter single-stpo-bm25
```

A standalone entry point is also available for ad-hoc runs; it reuses
the same judge factory and accepts a `--judge` flag with an Ollama
default:

```bash
python -m evaluation.end_to_end.run_all --judge ollama
```

## Output Files

One JSON file per configuration:



### Report Format


