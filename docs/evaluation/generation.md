# Generation Evaluation

**Module:** `evaluation/generation/`
**Metric:** Answer correctness (LLM-as-Judge)
**Test files:** `evaluation/tests/generation/test_judge.py`, `evaluation/tests/generation/test_metrics.py`, `evaluation/tests/generation/test_evaluate.py`

The generation evaluation measures how accurately the generator answers questions when provided with ground-truth context chunks and a variable number of distractor chunks. This tests both answer quality and robustness against retrieval noise.

---

## Evaluation Methodology

### Core Idea

For each answerable question in the evaluation dataset:

1. The ground-truth **relevant chunks** are always included in the context.
2. A variable number of **distractor chunks** (0 to 10) are added.
3. The generator produces an answer from this mixed context.
4. An **LLM-as-Judge** assesses whether the generated answer is semantically correct compared to the reference answer.

By varying the distractor count from 0 (pure gold context) to 10 (heavily diluted context), the evaluation measures how robust the generator is against retrieval noise.

### Distractor Selection

Distractors are selected **deterministically** using BM25 similarity:

1. All chunks not in `relevant_chunks` form the distractor pool.
2. A BM25 retriever indexes the pool and ranks chunks by similarity to the question.
3. The top-N most similar non-relevant chunks are selected as distractors.

This strategy produces the **hardest possible distractors** — chunks that are topically similar to the question but do not contain the answer. This simulates realistic retrieval noise more effectively than random selection.

**Determinism:** The same question + corpus always produces the same distractor ranking. Context order is shuffled with a fixed seed derived from the question ID.

### LLM-as-Judge

Answer correctness is assessed by prompting the same LLM as an evaluation judge:

- The judge receives the question, reference answer, and generated answer.
- It determines whether the generated answer conveys the same key information.
- Minor phrasing differences are acceptable; factual errors or missing core information are not.
- The judge returns a structured JSON response: `{correct, confidence, reasoning}`.

The judge is cleanly decoupled via an abstract `Judge` interface, allowing replacement with alternative evaluation strategies (e.g., RAGAS, human evaluation) without modifying the runner.

---

## Data Classes

### JudgementResult

```python
@dataclass
class JudgementResult:
    correct: bool        # Whether the answer is semantically correct
    confidence: float    # Judge's confidence (0.0–1.0)
    reasoning: str       # Explanation of the judgement
```

### GenerationEvalResult

```python
@dataclass
class GenerationEvalResult:
    question_id: str
    num_distractors: int
    correct: bool
    confidence: float
    generated_answer: str
    reference_answer: str
    judge_reasoning: str
    context_chunk_ids: list[str]
```

### GenerationMetrics

```python
@dataclass
class GenerationMetrics:
    accuracy: float                          # Overall correctness rate
    accuracy_by_distractors: dict[int, float]  # {0: 0.95, 1: 0.92, ...}
    num_queries: int
    knowledge_base: str
    model: str
    judge_model: str
```

---

## Usage

### Programmatic

```python
from src.marley.generator import OllamaGenerator
from src.marley.retrieval import load_chunks
from evaluation.generation.judge import OllamaJudge
from evaluation.generation.evaluate import run_and_report

chunks = load_chunks("data/chunks/stpo-chunks.json")
generator = OllamaGenerator(model="llama3.1:latest")
judge = OllamaJudge(model="llama3.1:latest")

report = run_and_report(
    generator, judge, chunks,
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
| `run_generation_evaluation(generator, judge, corpus, questions, levels)` | Run evaluation over all questions x levels. |
| `run_and_report(generator, judge, corpus, eval_path, levels)` | Full pipeline: load, run, aggregate, report. |

---

## Baseline Results

**Model:** `llama3.1:latest` (8B parameters, Ollama)
**Judge:** Same model as generator (LLM-as-Judge)

### StPO (75 questions, 151 chunks, overall accuracy: 0.497)

| Distractors | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.427 | 0.507 | 0.467 | 0.467 | 0.520 | 0.493 | 0.507 | 0.480 | 0.613 | 0.507 | 0.480 |

### FAQ-StPO (75 questions, 999 chunks, overall accuracy: 0.526)

| Distractors | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.587 | 0.480 | 0.547 | 0.587 | 0.520 | 0.507 | 0.493 | 0.507 | 0.520 | 0.507 | 0.533 |

### FAQ-AO (21 questions, 50 chunks, overall accuracy: 0.455)

| Distractors | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.524 | 0.429 | 0.524 | 0.429 | 0.381 | 0.476 | 0.429 | 0.429 | 0.524 | 0.381 | 0.476 |

### Analysis

The results show several notable patterns:

**1. No clear degradation with increasing distractors.** Contrary to the hypothesis that accuracy would decrease monotonically with more distractors, accuracy remains relatively stable across all distractor levels for all three knowledge bases. This suggests that the llama3.1:latest model is reasonably robust at identifying relevant information within mixed context, at least for the distractor counts tested (0-10).

**2. Overall accuracy around 50%.** All three knowledge bases show overall accuracy between 0.45 and 0.53. This moderate accuracy reflects the difficulty of the task: the model must extract precise regulatory information from context chunks that contain formal legal language.

**3. FAQ-StPO performs best (0.526).** The FAQ format — explicit question-answer pairs — provides clearer context for the generator than raw regulatory text. The FAQ-StPO corpus's structured Q/A format aligns well with the question-answering task.

**4. StPO shows surprising variance.** The StPO accuracy varies from 0.427 (0 distractors) to 0.613 (8 distractors). The lower accuracy at 0 distractors may indicate that some reference answers require information from multiple chunks, and the gold-standard context alone is insufficient.

**5. FAQ-AO has the lowest accuracy (0.455) despite the smallest corpus.** With only 21 evaluable questions, the sample size is small and results are more volatile (each question represents ~4.8% of the total).

**Limitations of LLM-as-Judge:** Using the same model for both generation and judgement introduces potential bias — the judge may be more lenient toward answers that match its own generation patterns. A stronger judge model or human evaluation would provide more reliable correctness assessments.

---

## Module Structure

```
evaluation/
├── __init__.py
├── generation/
│   ├── __init__.py
│   ├── judge.py            # Judge interface + OllamaJudge
│   ├── metrics.py          # GenerationEvalResult, GenerationMetrics, compute_generation_metrics()
│   └── evaluate.py         # Runner: select_distractors(), run_generation_evaluation(), run_and_report()
├── retrieval/
│   └── ...
└── tests/
    ├── __init__.py
    ├── generation/
    │   ├── __init__.py
    │   ├── test_judge.py    # 10 tests for judge parsing and OllamaJudge
    │   ├── test_metrics.py  # 6 tests for metric aggregation
    │   └── test_evaluate.py # 18 tests for distractor selection, context assembly, runner
    └── retrieval/
        └── ...
```
