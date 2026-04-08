# End-to-End Evaluation

**Module:** `evaluation/end_to_end/`
**Metric:** Full pipeline correctness across all configurations
**Test files:** `evaluation/tests/end_to_end/test_config.py`, `evaluation/tests/end_to_end/test_evaluate.py`, `evaluation/tests/end_to_end/test_prepare.py`, `evaluation/tests/end_to_end/test_metrics.py`

The end-to-end evaluation measures the complete MARley pipeline as a user would experience it: raw question in, answer (or abstention) out. No pre-selected chunks, no controlled distractors — just real retrieval feeding into real generation with real abstention.

**See also:** [Evaluation Overview](overview.md) | [Retrieval Evaluation](retrieval.md) | [Generation Evaluation](generation.md) | [Abstention Evaluation](abstention.md)

---

## Objective

Evaluate every pipeline configuration (retriever x knowledge base x combination strategy) on all 100 evaluation questions. Automatic abstention metrics are computed from the `expected_abstention` field in the evaluation dataset.

---

## Evaluation Methodology

### Pipeline Flow

```
Question
  |
  +-- [Single-KB] Retriever.retrieve(query, k=5)
  |   OR
  +-- [Merged Pool] MergedRetriever.retrieve(query, k=5)
  |   OR
  +-- [Fusion] FusionRetriever.retrieve(query, k=5)
  |       +-- Per-KB: retriever_i.retrieve(query, k=5)
  |       +-- rrf_fuse(all_results, k_rrf=60, k=5)
  |
  +-- normalize_scores(results, strategy)
  |
  +-- compute_confidence(normalized) -> confidence
  |
  +-- filter_by_threshold(normalized, threshold) -> filtered
  |
  +-- Level 1: len(filtered) == 0?
  |   +-- YES -> ABSTAIN (Level 1)
  |   +-- NO  -> continue
  |
  +-- Generator.generate(query, filtered) -> answer
  |
  +-- Level 2: detect_abstention(answer)?
  |   +-- YES -> ABSTAIN (Level 2)
  |   +-- NO  -> ANSWER
  |
  +-- E2EResult(answer, abstained, level, confidence, ...)
```

### Threshold Optimization

For each configuration, a Level 1 threshold sweep is performed before the full evaluation:

1. Retrieve and normalize scores for all 100 questions (no LLM calls).
2. Sweep thresholds from 0.0 to 1.0 in 0.05 steps.
3. At each threshold, compute abstention F1 from the Level 1 decisions.
4. Select the threshold that maximizes F1.

This ensures each configuration uses its own optimal abstention threshold.

---

## Configuration Matrix

### Single-KB Configurations (9)

| # | Name | Retriever | Knowledge Base | Strategy | Normalization |
|---|---|---|---|---|---|
| 1 | `single-stpo-bm25` | BM25 | StPO | single | bm25 |
| 2 | `single-stpo-vector` | Vector | StPO | single | vector |
| 3 | `single-stpo-hybrid` | Hybrid | StPO | single | rrf |
| 4 | `single-faq-stpo-bm25` | BM25 | FAQ-StPO | single | bm25 |
| 5 | `single-faq-stpo-vector` | Vector | FAQ-StPO | single | vector |
| 6 | `single-faq-stpo-hybrid` | Hybrid | FAQ-StPO | single | rrf |
| 7 | `single-faq-ao-bm25` | BM25 | FAQ-AO | single | bm25 |
| 8 | `single-faq-ao-vector` | Vector | FAQ-AO | single | vector |
| 9 | `single-faq-ao-hybrid` | Hybrid | FAQ-AO | single | rrf |

### Combined-KB Merged Pool Configurations (12)

| # | Name | Retriever | Combination | Strategy | Normalization |
|---|---|---|---|---|---|
| 10 | `merged-stpo+faq-stpo-bm25` | BM25 | StPO + FAQ-StPO | merged_pool | bm25 |
| 11 | `merged-stpo+faq-stpo-vector` | Vector | StPO + FAQ-StPO | merged_pool | vector |
| 12 | `merged-stpo+faq-stpo-hybrid` | Hybrid | StPO + FAQ-StPO | merged_pool | rrf |
| 13 | `merged-stpo+faq-ao-bm25` | BM25 | StPO + FAQ-AO | merged_pool | bm25 |
| 14 | `merged-stpo+faq-ao-vector` | Vector | StPO + FAQ-AO | merged_pool | vector |
| 15 | `merged-stpo+faq-ao-hybrid` | Hybrid | StPO + FAQ-AO | merged_pool | rrf |
| 16 | `merged-faq-stpo+faq-ao-bm25` | BM25 | FAQ-StPO + FAQ-AO | merged_pool | bm25 |
| 17 | `merged-faq-stpo+faq-ao-vector` | Vector | FAQ-StPO + FAQ-AO | merged_pool | vector |
| 18 | `merged-faq-stpo+faq-ao-hybrid` | Hybrid | FAQ-StPO + FAQ-AO | merged_pool | rrf |
| 19 | `merged-all-bm25` | BM25 | StPO + FAQ-StPO + FAQ-AO | merged_pool | bm25 |
| 20 | `merged-all-vector` | Vector | StPO + FAQ-StPO + FAQ-AO | merged_pool | vector |
| 21 | `merged-all-hybrid` | Hybrid | StPO + FAQ-StPO + FAQ-AO | merged_pool | rrf |

### Combined-KB Fusion Configurations (12)

| # | Name | Retriever | Combination | Strategy | Normalization |
|---|---|---|---|---|---|
| 22 | `fusion-stpo+faq-stpo-bm25` | BM25 | StPO + FAQ-StPO | fusion | rrf |
| 23 | `fusion-stpo+faq-stpo-vector` | Vector | StPO + FAQ-StPO | fusion | rrf |
| 24 | `fusion-stpo+faq-stpo-hybrid` | Hybrid | StPO + FAQ-StPO | fusion | rrf |
| 25 | `fusion-stpo+faq-ao-bm25` | BM25 | StPO + FAQ-AO | fusion | rrf |
| 26 | `fusion-stpo+faq-ao-vector` | Vector | StPO + FAQ-AO | fusion | rrf |
| 27 | `fusion-stpo+faq-ao-hybrid` | Hybrid | StPO + FAQ-AO | fusion | rrf |
| 28 | `fusion-faq-stpo+faq-ao-bm25` | BM25 | FAQ-StPO + FAQ-AO | fusion | rrf |
| 29 | `fusion-faq-stpo+faq-ao-vector` | Vector | FAQ-StPO + FAQ-AO | fusion | rrf |
| 30 | `fusion-faq-stpo+faq-ao-hybrid` | Hybrid | FAQ-StPO + FAQ-AO | fusion | rrf |
| 31 | `fusion-all-bm25` | BM25 | StPO + FAQ-StPO + FAQ-AO | fusion | rrf |
| 32 | `fusion-all-vector` | Vector | StPO + FAQ-StPO + FAQ-AO | fusion | rrf |
| 33 | `fusion-all-hybrid` | Hybrid | StPO + FAQ-StPO + FAQ-AO | fusion | rrf |

**Total: 33 configurations x 100 questions = 3,300 pipeline runs.**

**Note on fusion normalization:** All fusion configurations use `rrf` normalization because RRF fusion produces RRF scores regardless of the underlying per-KB retriever type.

---

## Data Classes

### E2EConfig

```python
@dataclass(frozen=True)
class E2EConfig:
    name: str                       # Unique config name
    retriever_type: str             # "bm25", "vector", "hybrid"
    knowledge_bases: tuple[str, ...]  # KB identifiers
    strategy: str                   # "single", "merged_pool", "fusion"
    normalization_strategy: str     # "bm25", "vector", "rrf"
    k: int = 5                      # Top-k retrieval results
    k_rrf: int = 60                 # RRF smoothing constant
```

### E2EResult

```python
@dataclass
class E2EResult:
    question_id: str
    question: str
    reference_answer: str
    category: str                   # "direct", "multi_source", "unanswerable"
    expected_abstention: bool
    answer: str                     # Generated answer (empty if abstained)
    abstained: bool
    abstention_level: int | None    # 1 = retrieval, 2 = LLM, None = answered
    abstention_reason: str
    confidence: float               # Top-1 normalized retrieval score
    retrieval_chunk_ids: list[str]  # Chunk IDs in retrieved context
    model: str                      # Generator model name
```

### FusionRetriever

```python
class FusionRetriever(Retriever):
    """Wraps multiple pre-indexed retrievers with RRF fusion."""

    def __init__(self, retrievers: list[Retriever], k_rrf: int = 60) -> None: ...
    def index(self, corpus) -> None: ...  # raises NotImplementedError
    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]: ...
    def size(self) -> int: ...  # sum of sub-retriever sizes
```

### E2EConfigMetrics

```python
@dataclass
class E2EConfigMetrics:
    config_name: str
    num_total: int
    num_abstained: int
    abstention_rate: float
    abstention_precision: float       # correct_abs / (correct_abs + incorrect_abs)
    abstention_recall: float          # correct_abs / (correct_abs + missing_abs)
    abstention_f1: float
    abstention_by_category: dict[str, dict]   # per-category breakdown
    avg_confidence: float
    level1_abstentions: int
    level2_abstentions: int
```

Abstention categories (derived from `expected_abstention` flag, no human labels needed):
- **Correct abstention:** `expected_abstention=True` AND `abstained=True`
- **Incorrect abstention:** `expected_abstention=False` AND `abstained=True`
- **Missing abstention:** `expected_abstention=True` AND `abstained=False`

---

## Functions

| Function | Module | Description |
|---|---|---|
| `generate_all_configs()` | `config.py` | Generate all 33 E2E configurations |
| `load_questions()` | `evaluate.py` | Load questions from evaluation.json |
| `sweep_threshold()` | `evaluate.py` | Sweep Level 1 thresholds (no LLM) |
| `run_e2e_config()` | `evaluate.py` | Run full pipeline for one configuration |
| `run_and_report()` | `evaluate.py` | Sweep + run + report in one call |
| `save_report()` | `evaluate.py` | Save report to JSON |
| `compute_e2e_config_metrics()` | `metrics.py` | Compute per-config automatic metrics |
| `build_comparison_table()` | `metrics.py` | Build cross-config comparison table |

---

## Usage

### Running a Single Configuration

```python
from evaluation.end_to_end.config import generate_all_configs
from evaluation.end_to_end.evaluate import load_questions, run_and_report, save_report

configs = generate_all_configs()
questions = load_questions("data/testing/evaluation.json")

# Set up retriever (example: single-KB BM25)
retriever = BM25Retriever()
retriever.index(chunks)

report = run_and_report(configs[0], retriever, generator, questions)
save_report(report, "results/e2e/single-stpo-bm25.json")
```

### Computing Comparison Metrics

```python
import json
from dataclasses import asdict
from pathlib import Path
from evaluation.end_to_end.evaluate import E2EResult
from evaluation.end_to_end.metrics import build_comparison_table, compute_e2e_config_metrics

all_metrics = []
for path in sorted(Path("results/e2e").glob("e2e-results-*.json")):
    report = json.loads(path.read_text())
    results = [E2EResult(**r) for r in report["results"]]
    config_name = report["config"]["name"]
    m = compute_e2e_config_metrics(results, config_name)
    all_metrics.append(m)

table = build_comparison_table(all_metrics)
```

---

## Results

*To be populated after evaluation runs.*

---

## Module Structure

```
evaluation/end_to_end/
  __init__.py
  config.py           # E2EConfig, generate_all_configs()
  evaluate.py         # E2EResult, load_questions(), sweep_threshold(),
                      # run_e2e_config(), run_and_report(), save_report()
  metrics.py          # E2EConfigMetrics, compute_e2e_config_metrics(),
                      # build_comparison_table()
  run_all.py          # CLI: run all 33 configurations

src/marley/retrieval/
  fusion.py           # FusionRetriever (added alongside rrf_fuse)
```
