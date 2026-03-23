# Abstention Evaluation

**Module:** `evaluation/abstention/`
**Metric:** Abstention quality (precision, recall, F1)
**Test files:** `evaluation/tests/abstention/test_metrics.py`, `evaluation/tests/abstention/test_evaluate.py`

The abstention evaluation measures how well the system distinguishes answerable from unanswerable questions using a two-level abstention mechanism. Level 1 operates on retrieval confidence scores (no LLM required), while Level 2 detects abstentions produced by the language model during generation.

**See also:** [Evaluation Overview](overview.md) | [Abstention Pipeline](../abstention/abstention.md)

---

## Objective

The goal is to evaluate the system's ability to correctly refuse unanswerable questions while still answering those it can. A well-calibrated abstention mechanism should:

1. **Abstain** when the question falls outside the knowledge base scope (high recall).
2. **Not abstain** when the question is answerable (low false abstention rate).
3. **Balance** both objectives, measured via the F1 score.

The two-level design allows Level 1 to catch clearly unanswerable questions cheaply (no LLM call), while Level 2 provides a safety net for borderline cases that pass Level 1 but still cannot be answered from the retrieved context.

---

## Methodology

### Level 1: Threshold Sweep

Level 1 abstention is **automatic and requires no LLM**. It operates on normalized retrieval confidence scores:

1. For each question, the retriever returns ranked results with scores.
2. The top score is normalized to [0, 1] using a retriever-specific normalization strategy.
3. If the normalized score falls below a threshold, the system abstains (Level 1 abstention).

The **threshold sweep** evaluates all thresholds in [0.0, 0.05, 0.10, ..., 1.0] to find the optimal operating point:

- Retrieval scores are **pre-computed once** for all questions.
- For each threshold, the sweep determines which questions trigger Level 1 abstention.
- Predictions are compared against the `expected_abstention` ground truth from the evaluation dataset.
- Abstention metrics (precision, recall, F1) are computed at each threshold.

This design makes the sweep **fast**: no LLM calls, no generation, just score comparisons across 21 threshold values.

### Level 2: Full Evaluation

At the best Level 1 threshold (selected from the sweep), the full pipeline runs:

1. Questions below the threshold are abstained at Level 1 (no generation).
2. Questions above the threshold proceed to generation.
3. The generated answer is inspected for Level 2 (LLM) abstention markers — phrases indicating the model could not answer from the provided context.
4. Both levels are combined to produce the final abstention decision.

This provides the complete picture: Level 1 efficiency plus Level 2 coverage.

---

## Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Abstention Precision** | correct_abstention / (correct + incorrect) | Of all abstentions, how many were justified? |
| **Abstention Recall** | correct_abstention / (correct + missing) | Of all unanswerable questions, how many did we abstain on? |
| **F1** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| **False Abstention Rate** | incorrect / answerable | How often the system incorrectly refuses an answerable question |
| **Coverage** | answered / total | Proportion of questions that receive an answer |

- **correct_abstention**: Unanswerable questions where the system abstained (true positive).
- **incorrect**: Answerable questions where the system abstained (false positive).
- **missing**: Unanswerable questions where the system failed to abstain (false negative).

---

## Level 1 Sweep

The `run_level1_sweep()` function pre-computes retrieval scores once, then sweeps all thresholds without any LLM calls.

**Scope:** 3 retrievers x 3 KBs x 21 thresholds x 100 questions = 18,900 evaluations

For each retriever-KB combination, the sweep produces a table of metrics across thresholds, enabling selection of the optimal threshold that maximizes F1 (or any other target metric).

---

## Level 2 Evaluation

The `run_abstention_evaluation()` function runs the full two-level pipeline at a selected threshold.

**Scope:** 3 retrievers x 3 KBs x 100 questions = 900 pipeline runs

Each run involves retrieval, threshold checking, and (if above threshold) generation. The evaluation captures both Level 1 and Level 2 abstention decisions and computes the combined metrics.

---

## Usage

### Programmatic

```python
from src.marley.retrieval import BM25Retriever, load_chunks
from evaluation.abstention.evaluate import run_level1_sweep, run_abstention_evaluation

chunks = load_chunks("data/chunks/stpo-chunks.json")
retriever = BM25Retriever()

# Level 1 sweep
sweep = run_level1_sweep(
    retriever, chunks, questions,
    thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    normalization_strategy="bm25",
)

# Full evaluation
report = run_abstention_evaluation(
    retriever, generator, chunks, questions,
    threshold=0.3, normalization_strategy="bm25",
)
```

### Functions

| Function | Description |
|---|---|
| `run_level1_sweep()` | Sweep thresholds, Level 1 only. |
| `run_abstention_evaluation()` | Full two-level evaluation at a given threshold. |
| `run_and_report()` | Complete pipeline with sweep + evaluation. |
| `compute_abstention_metrics()` | Compute precision, recall, F1 from results. |

---

## Module Structure

```
evaluation/
├── abstention/
│   ├── __init__.py
│   ├── metrics.py          # AbstentionMetrics + compute_abstention_metrics()
│   └── evaluate.py         # Level 1 sweep + Level 2 runner
└── tests/
    └── abstention/
        ├── __init__.py
        ├── test_metrics.py  # 10 tests for metric computation
        └── test_evaluate.py # 12 tests for sweep and evaluation runner
```
