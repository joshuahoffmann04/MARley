# Combined Knowledge Base Generation Evaluation

**Module:** `evaluation/generation/combined.py`
**Shared utilities:** `evaluation/utils.py`
**Test file:** `evaluation/tests/generation/test_combined.py`

This evaluation measures whether providing context from multiple knowledge bases improves generation quality compared to single-KB generation.

**See also:** [Generation Evaluation](generation.md) | [Combined-KB Retrieval](combined-retrieval.md) | [Manual Evaluation](manual-evaluation.md)

---

## Objective

The thesis proposal (Goal G3) requires evaluating answer generation quality across different retrieval configurations. The single-KB generation evaluation tests distractor robustness on individual knowledge bases. This module extends that to combined-KB configurations: does providing relevant context from multiple sources improve the generator's answers?

---

## Methodology

### Controlled Evaluation (Matching Single-KB Approach)

The combined-KB generation evaluation uses the **same controlled methodology** as the single-KB evaluation:

1. **Ground-truth relevant chunks** from all included KBs are always present in the context.
2. A variable number of **BM25-ranked distractor chunks** (0 to 10) are added from the merged corpus.
3. The generator produces an answer from this mixed context.
4. Correctness is assessed separately via the [Manual Evaluation Framework](manual-evaluation.md).

This is a deliberate design choice. By using the same methodology, combined-KB results are **directly comparable** with single-KB baselines. The only variable that changes is the source configuration:

| Aspect | Single-KB | Combined-KB |
|---|---|---|
| Relevant chunks | From one KB | Set union across multiple KBs |
| Distractor pool | Single KB corpus | Merged corpus (all included KBs) |
| Distractor ranking | BM25 similarity | BM25 similarity (same algorithm) |
| Context assembly | Relevant + distractors, shuffled | Relevant + distractors, shuffled |
| Generator | Same model | Same model |
| Distractor levels | 0–10 | 0–10 |

### Why Not Actual Retrieval?

An alternative approach would use actual combined retrieval (best strategy from Phase 1) to obtain context, then generate answers from retrieved chunks. This was deliberately not chosen because:

1. **Confounded variables:** Using actual retrieval conflates retrieval quality with generation quality. If answers improve, is it because the generator benefits from multi-source context, or because the retriever happens to find better chunks?
2. **No distractor-level analysis:** Actual retrieval produces a fixed set of chunks per query. The distractor-level dimension (0–10) would be lost, eliminating the noise-robustness analysis.
3. **No direct comparison:** Single-KB generation uses controlled context. Using a different methodology for combined-KB would prevent fair comparison.

The controlled approach isolates the generation effect: "Given that the correct chunks are available from multiple KBs, does the generator produce better answers?"

### Merging Strategy

For each question, relevant chunks are merged across all included KBs using **set union**:

```
question eval-001:
  evaluation-stpo.json:      relevant_chunks = ["par-7-txt-1"]
  evaluation-faq-stpo.json:  relevant_chunks = ["faq-stpo-stpo-0012"]
  merged:                    relevant_chunks = ["par-7-txt-1", "faq-stpo-stpo-0012"]
```

This means the generator receives **more relevant context** for combined-KB evaluations — chunks from different KBs that each address the question from a different angle (e.g., formal regulation text from StPO + FAQ explanation from FAQ-StPO). The hypothesis is that this multi-source context improves answer quality.

### Distractor Selection

Distractors are selected from the **merged corpus** (all chunks from all included KBs). BM25 ranking ensures the most confusing non-relevant chunks are chosen first, consistent with the single-KB approach. The larger merged corpus provides a harder distractor pool, since there are more topically similar but non-relevant chunks available.

---

## KB Combinations

Same 4 combinations as the [Combined-KB Retrieval Evaluation](combined-retrieval.md):

| ID | Combination | Chunks | Evaluable Queries |
|---|---|---|---|
| stpo+faq-stpo | StPO + FAQ-StPO | 153 + 1039 = 1192 | 75 |
| stpo+faq-ao | StPO + FAQ-AO | 153 + 0 = 153 | 75 |
| faq-stpo+faq-ao | FAQ-StPO + FAQ-AO | 1039 + 0 = 1039 | 75 |
| all | StPO + FAQ-StPO + FAQ-AO | 153 + 1039 + 0 = 1192 | 75 |

All 4 combinations evaluate the same 75 non-unanswerable questions (questions with `expected_abstention: true` are skipped). With distractor levels 0–10, each combination produces **75 × 11 = 825 results**, for a total of **3,300 results** across all combinations.

---

## Evaluation Data

For each question, relevant chunks from all included KBs are merged using set union. A question is evaluable if it has at least one relevant chunk across any included KB. This yields **75 evaluable queries** for all combinations.

The merge utilities are shared between retrieval and generation evaluation modules via `evaluation/utils.py`:

- `merge_chunks(*chunk_paths)` — loads and concatenates chunk corpora, validates no duplicate chunk IDs
- `merge_evaluation_data(eval_paths)` — merges question evaluation data with set union of relevant chunks

---

## Data Classes

The combined generation evaluation reuses the same data classes as the single-KB evaluation:

### GenerationEvalResult

```python
@dataclass
class GenerationEvalResult:
    question_id: str
    num_distractors: int
    generated_answer: str
    reference_answer: str
    context_chunk_ids: list[str]
```

### GenerationMetrics

```python
@dataclass
class GenerationMetrics:
    num_results: int
    results_by_distractors: dict[int, int]
    num_queries: int
    knowledge_base: str  # Contains the combination name (e.g., "stpo+faq-stpo")
    model: str
```

---

## Functions

| Function | Description |
|---|---|
| `run_combined_generation_evaluation(generator, chunk_paths, eval_paths, levels)` | Merge data, run generation evaluation across all questions × levels. |
| `run_and_report_combined(generator, chunk_paths, eval_paths, levels, combination_name)` | Full pipeline: merge, run, aggregate metrics, return report dict. |

---

## Usage

### Run a Single Combination

```python
from src.marley.generator import OllamaGenerator
from evaluation.generation.combined import run_and_report_combined

generator = OllamaGenerator(model="llama3.1:latest")

report = run_and_report_combined(
    generator,
    chunk_paths={
        "stpo": "data/chunks/stpo-chunks.json",
        "faq-stpo": "data/chunks/faq-stpo-chunks.json",
    },
    eval_paths={
        "stpo": "data/testing/evaluation-stpo.json",
        "faq-stpo": "data/testing/evaluation-faq-stpo.json",
    },
    distractor_levels=[0, 1, 3, 5, 10],
)
print(report["metrics"])
```

### Run All 4 Combinations

```python
from src.marley.generator import OllamaGenerator
from evaluation.generation.combined import run_and_report_combined
import json
from pathlib import Path

generator = OllamaGenerator(model="llama3.1:latest")

CHUNK_PATHS = {
    "stpo": "data/chunks/stpo-chunks.json",
    "faq-stpo": "data/chunks/faq-stpo-chunks.json",
    "faq-ao": "data/chunks/faq-ao-chunks.json",
}

EVAL_PATHS = {
    "stpo": "data/testing/evaluation-stpo.json",
    "faq-stpo": "data/testing/evaluation-faq-stpo.json",
    "faq-ao": "data/testing/evaluation-faq-ao.json",
}

COMBINATIONS = {
    "stpo+faq-stpo": ["stpo", "faq-stpo"],
    "stpo+faq-ao": ["stpo", "faq-ao"],
    "faq-stpo+faq-ao": ["faq-stpo", "faq-ao"],
    "all": ["stpo", "faq-stpo", "faq-ao"],
}

for combo_name, kb_list in COMBINATIONS.items():
    report = run_and_report_combined(
        generator,
        chunk_paths={kb: CHUNK_PATHS[kb] for kb in kb_list},
        eval_paths={kb: EVAL_PATHS[kb] for kb in kb_list},
        combination_name=combo_name,
    )
    out = Path(f"data/testing/generation-eval-combined-{combo_name}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{combo_name}: {report['metrics']['num_results']} results")
```

### Prepare Manual Evaluation Items

After generation runs complete, prepare items for the manual evaluation UI:

```python
from evaluation.manual.prepare import prepare_generation_items
from evaluation.manual.models import save_items

for combo in ["stpo+faq-stpo", "stpo+faq-ao", "faq-stpo+faq-ao", "all"]:
    items = prepare_generation_items(
        f"data/testing/generation-eval-combined-{combo}.json",
        knowledge_base=combo,
    )
    save_items(items, f"data/testing/manual-eval-items-combined-{combo}.json")
    print(f"{combo}: {len(items)} items prepared")
```

---

## Results

*Results will be populated after evaluation runs are completed.*

### Expected Output

Each combination produces a result file `generation-eval-combined-{combination}.json` with the same structure as the single-KB result files:

```json
{
    "combination": "stpo+faq-stpo",
    "eval_files": {"stpo": "...", "faq-stpo": "..."},
    "config": {
        "distractor_levels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "generator_model": "llama3.1:latest",
        "corpus_size": 1192,
        "knowledge_bases": ["faq-stpo", "stpo"],
        "combination": "stpo+faq-stpo"
    },
    "metrics": { ... },
    "results": [ ... ]
}
```

Correctness assessment is performed via the [Manual Evaluation Framework](manual-evaluation.md).

---

## Module Structure

```
evaluation/
├── utils.py                        # Shared merge utilities
│                                   #   merge_chunks(), merge_evaluation_data()
├── generation/
│   ├── __init__.py                 # Exports single-KB and combined-KB functions
│   ├── evaluate.py                 # Single-KB generation runner
│   ├── metrics.py                  # GenerationEvalResult, GenerationMetrics
│   └── combined.py                 # Combined-KB generation runner (this module)
│
└── tests/
    └── generation/
        ├── test_evaluate.py        # 17 tests for single-KB generation
        ├── test_metrics.py         # 5 tests for metric aggregation
        └── test_combined.py        # 14 tests for combined-KB generation
```
