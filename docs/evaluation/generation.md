# Generation Evaluation

**Module:** `evaluation/generation/`
**Test files:** `evaluation/tests/generation/`

The generation evaluation measures how well the generator produces answers when provided
with ground-truth context chunks and a variable number of distractor chunks. Quality is
assessed automatically via ROUGE, BERTScore, and an optional LLM judge.

**See also:** [LLM Judge](judge.md) | [Evaluation Overview](overview.md) | [Combined-KB Generation](combined-generation.md) | [Abstention Evaluation](abstention.md)

---

## Evaluation Methodology

### Core Idea

For each answerable question in the evaluation dataset:

1. The ground-truth **relevant chunks** are always included in the context.
2. A variable number of **distractor chunks** (0 to 10) are added.
3. The generator produces an answer from this mixed context.
4. Quality scores are computed automatically.

By varying the distractor count from 0 (pure gold context) to 10 (heavily diluted context),
the evaluation produces answers under increasingly noisy conditions, revealing how retrieval
noise degrades generation quality.

### Distractor Selection

Distractors are selected **deterministically** using BM25 similarity:

1. All chunks not in `relevant_chunks` form the distractor pool.
2. A BM25 retriever indexes the pool and ranks chunks by similarity to the question.
3. The top-N most similar non-relevant chunks are selected as distractors.

This produces the **hardest possible distractors** — chunks topically similar to the question
but not containing the answer — simulating realistic retrieval noise.

Context order is shuffled with a fixed seed derived from the question ID, ensuring
reproducibility across evaluation runs.

---

## Quality Metrics

### Automatic Text-Similarity Metrics (HuggingFace `evaluate`)

| Metric | Type | What it measures |
|---|---|---|
| **ROUGE-1** | N-gram overlap | Unigram F1 between generated and reference answer |
| **ROUGE-2** | N-gram overlap | Bigram F1 between generated and reference answer |
| **ROUGE-L** | N-gram overlap | Longest-common-subsequence F1 |
| **BERTScore F1** | Semantic similarity | BERT-embedding cosine similarity between generated and reference |

ROUGE is deterministic and fast. BERTScore is model-based and more robust to
paraphrasing — it captures semantic equivalence that pure n-gram metrics miss.

Both are computed in **batch** over all results to minimise model loading overhead.

### LLM Judge Scores (optional)

When a `Judge` instance is passed to the evaluation runner, three additional scores
are computed per answer using a second LLM call:

| Score | What it measures |
|---|---|
| **faithfulness** | Does the answer only use information from the provided context? |
| **answer_relevance** | Does the answer address the question asked? |
| **correctness** | Does the answer agree with the reference answer? |

All scores are in [0.0, 1.0]. See [LLM Judge](judge.md) for architecture and usage.

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
    # ROUGE (n-gram overlap)
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    # BERTScore (semantic similarity)
    bertscore_f1: float = 0.0
    # LLM judge scores
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    correctness: float = 0.0
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
    # Macro-averaged quality scores
    avg_rouge1: float = 0.0
    avg_rouge2: float = 0.0
    avg_rougeL: float = 0.0
    avg_bertscore_f1: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_correctness: float = 0.0
```

---

## Usage

### Without judge (ROUGE + BERTScore only)

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

### With LLM judge (full metrics)

```python
from evaluation.judge import OllamaJudge

judge = OllamaJudge(model="llama3.1:latest")

report = run_and_report(
    generator, chunks,
    "data/testing/evaluation-stpo.json",
    distractor_levels=[0, 5, 10],
    knowledge_base="stpo",
    judge=judge,
)
# report["metrics"] now includes avg_faithfulness, avg_answer_relevance, avg_correctness
```

### Functions

| Function | Description |
|---|---|
| `select_distractors(question, relevant_ids, corpus, max)` | BM25-ranked distractor selection. |
| `run_generation_evaluation(generator, corpus, questions, levels, judge)` | Run evaluation over all questions × levels. |
| `run_and_report(generator, corpus, eval_path, levels, judge)` | Full pipeline: load, run, aggregate, report. |

---

## Module Structure

```
evaluation/
├── generation/
│   ├── __init__.py             # Exports for single-KB and combined-KB functions
│   ├── metrics.py              # GenerationEvalResult, GenerationMetrics, compute_generation_metrics()
│   ├── hf_metrics.py           # ROUGE + BERTScore via HuggingFace evaluate
│   ├── evaluate.py             # Single-KB runner: select_distractors(), run_generation_evaluation()
│   └── combined.py             # Combined-KB runner (see combined-generation.md)
├── judge/                      # LLM judge module (see judge.md)
│   ├── base.py                 # Judge ABC + JudgementResult
│   ├── prompts.py              # Judge prompt templates
│   ├── ollama_judge.py         # OllamaJudge
│   └── openai_judge.py         # OpenAIJudge
└── tests/
    └── generation/
        ├── test_metrics.py     # 11 tests for metric aggregation + quality fields
        ├── test_evaluate.py    # 22 tests for single-KB generation + judge integration
        ├── test_combined.py    # 14 tests for combined-KB generation
        └── test_hf_metrics.py  # 11 tests for ROUGE + BERTScore helpers
```
