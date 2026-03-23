# Generation Evaluation

**Module:** `evaluation/generation/`
**Metric:** Answer generation with distractor robustness testing
**Test files:** `evaluation/tests/generation/test_metrics.py`, `evaluation/tests/generation/test_evaluate.py`

The generation evaluation measures how the generator produces answers when provided with ground-truth context chunks and a variable number of distractor chunks. This tests answer generation under varying levels of retrieval noise.

**Correctness assessment** is performed separately via the [Manual Evaluation Framework](manual-evaluation.md).

**See also:** [Evaluation Overview](overview.md) | [Combined-KB Generation](combined-generation.md) | [Manual Evaluation](manual-evaluation.md) | [Abstention Evaluation](abstention.md)

---

## Evaluation Methodology

### Core Idea

For each answerable question in the evaluation dataset:

1. The ground-truth **relevant chunks** are always included in the context.
2. A variable number of **distractor chunks** (0 to 10) are added.
3. The generator produces an answer from this mixed context.

By varying the distractor count from 0 (pure gold context) to 10 (heavily diluted context), the evaluation produces answers under increasingly noisy conditions. Correctness is then assessed via the manual evaluation UI.

### Distractor Selection

Distractors are selected **deterministically** using BM25 similarity:

1. All chunks not in `relevant_chunks` form the distractor pool.
2. A BM25 retriever indexes the pool and ranks chunks by similarity to the question.
3. The top-N most similar non-relevant chunks are selected as distractors.

This strategy produces the **hardest possible distractors** — chunks that are topically similar to the question but do not contain the answer. This simulates realistic retrieval noise more effectively than random selection.

**Determinism:** The same question + corpus always produces the same distractor ranking. Context order is shuffled with a fixed seed derived from the question ID.

---

## Data Classes

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
    knowledge_base: str
    model: str
```

---

## Usage

### Programmatic

```python
from src.marley.generator import OllamaGenerator
from src.marley.retrieval import load_chunks
from evaluation.generation.evaluate import run_and_report

chunks = load_chunks("data/chunks/stpo-chunks.json")
generator = OllamaGenerator(model="llama3.1:latest")

report = run_and_report(
    generator, chunks,
    "data/testing/evaluation-stpo.json",
    distractor_levels=[0, 1, 3, 5, 10],
    knowledge_base="stpo",
)
print(report["metrics"])
```

### Functions

| Function | Description |
|---|---|
| `select_distractors(question, relevant_ids, corpus, max)` | BM25-ranked distractor selection. |
| `run_generation_evaluation(generator, corpus, questions, levels)` | Run evaluation over all questions x levels. |
| `run_and_report(generator, corpus, eval_path, levels)` | Full pipeline: load, run, aggregate, report. |

---

## Baseline Results (LLM-as-Judge — Historical)

The following results were obtained using an LLM-as-Judge (llama3.1:latest) for automated correctness assessment. These are **superseded by manual evaluation** which provides more reliable and defensible correctness assessments.

**Model:** `llama3.1:latest` (8B parameters, Ollama)
**Judge:** Same model as generator (LLM-as-Judge — archived)

### StPO (75 questions, 153 chunks, overall accuracy: 0.497)

| Distractors | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.427 | 0.507 | 0.467 | 0.467 | 0.520 | 0.493 | 0.507 | 0.480 | 0.613 | 0.507 | 0.480 |

### FAQ-StPO (75 questions, 1039 chunks, overall accuracy: 0.526)

| Distractors | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.587 | 0.480 | 0.547 | 0.587 | 0.520 | 0.507 | 0.493 | 0.507 | 0.520 | 0.507 | 0.533 |

### FAQ-AO (21 questions, 0 chunks — pending)

| Distractors | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.524 | 0.429 | 0.524 | 0.429 | 0.381 | 0.476 | 0.429 | 0.429 | 0.524 | 0.381 | 0.476 |

### Analysis

**Limitation:** Using the same model for both generation and judgement introduces potential bias — the judge may be more lenient toward answers matching its own generation patterns. Manual evaluation provides more reliable correctness assessments and supersedes these results.

---

## Module Structure

```
evaluation/
├── __init__.py
├── utils.py                    # Shared merge utilities (merge_chunks, merge_evaluation_data)
├── generation/
│   ├── __init__.py             # Exports single-KB and combined-KB functions
│   ├── metrics.py              # GenerationEvalResult, GenerationMetrics, compute_generation_metrics()
│   ├── evaluate.py             # Single-KB runner: select_distractors(), run_generation_evaluation()
│   └── combined.py             # Combined-KB runner (see combined-generation.md)
├── manual/                     # Manual evaluation framework (see manual-evaluation.md)
│   └── ...
└── tests/
    ├── __init__.py
    ├── generation/
    │   ├── __init__.py
    │   ├── test_metrics.py     # 5 tests for metric aggregation
    │   ├── test_evaluate.py    # 17 tests for single-KB generation
    │   └── test_combined.py    # 14 tests for combined-KB generation
    └── manual/
        └── ...
```
