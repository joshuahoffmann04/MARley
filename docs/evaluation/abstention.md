# Abstention Evaluation

> Evaluates the two-level abstention mechanism: retrieval confidence threshold (Level 1) and LLM-based detection (Level 2).

## Two-Level Abstention

The MARley pipeline uses a two-level abstention system:

1. **Level 1 -- Retrieval Confidence**: Before generation, retrieval scores are normalized and a confidence score is computed. If confidence falls below a threshold, the system abstains without calling the LLM.
2. **Level 2 -- LLM Detection**: If Level 1 passes, the LLM generates an answer. If the answer contains an `ABSTENTION:` prefix, the system detects and extracts the abstention reason.

## Metrics

Five metrics evaluate abstention quality. These are shared between the abstention evaluation and the end-to-end evaluation via the `AbstentionMetrics` dataclass in `evaluation/utils.py`.

### Precision

```
Precision = correct_abstention / (correct_abstention + incorrect_abstention)
```

When the system decides to abstain, how often is it correct? Returns 1.0 if there are no abstentions (vacuous truth).

### Recall

```
Recall = correct_abstention / (correct_abstention + missing_abstention)
```

Of all truly unanswerable questions, how many does the system correctly identify? Returns 1.0 if there are no unanswerable questions.

### F1

```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Harmonic mean of precision and recall. The primary optimization target for threshold selection.

### False Abstention Rate

```
False Abstention Rate = incorrect_abstention / (answered + incorrect_abstention)
```

Of all answerable questions, how many does the system wrongly refuse to answer? Lower is better.

### Coverage

```
Coverage = (answered + missing_abstention) / total
```

Proportion of questions that receive an answer (correct or not). A system that never abstains has Coverage = 1.0.

## Implementation

### AbstentionMetrics Dataclass

```python
@dataclass
class AbstentionMetrics:
    precision: float
    recall: float
    f1: float
    false_abstention_rate: float
    coverage: float
    num_correct_abstention: int
    num_incorrect_abstention: int
    num_missing_abstention: int
    num_answered: int
    num_total: int
    threshold: float
```

Location: `evaluation/utils.py`

### compute_abstention_metrics()

```python
def compute_abstention_metrics(
    results: list[dict],
    threshold: float,
) -> AbstentionMetrics:
```

Input: list of dicts with `expected_abstention` (bool) and `system_abstained` (bool).

The four-way classification:

| Expected \ System | Abstained | Answered |
|---|---|---|
| **Unanswerable** | correct_abstention | missing_abstention |
| **Answerable** | incorrect_abstention | answered |

## Evaluation Modes

### Level 1 Sweep (Fast, No LLM)

Sweeps threshold values to find the optimal Level 1 threshold using only retrieval scores.

```python
def run_level1_sweep(
    retriever: Retriever,
    corpus: list[dict],
    questions: list[dict],
    thresholds: list[float],
    *,
    k: int = 5,
    normalization_strategy: str = "vector",
    normalization_params: dict | None = None,
) -> list[dict]:
```

**Process**:
1. Index the retriever on the corpus
2. For each question: retrieve, normalize scores, compute confidence
3. For each threshold: determine which questions would trigger Level 1 abstention
4. Compute abstention metrics at each threshold

**Default thresholds**: 0.0 to 1.0 in 0.05 steps (21 values)

**Output**: List of `{threshold, metrics}` dicts

Implementation: `evaluation/abstention/evaluate.py` (`run_level1_sweep()`)

### Full Two-Level Evaluation

Runs the complete abstention pipeline including LLM generation.

```python
def run_abstention_evaluation(
    retriever: Retriever,
    generator: Generator,
    corpus: list[dict],
    questions: list[dict],
    *,
    k: int = 5,
    threshold: float = 0.3,
    normalization_strategy: str = "vector",
    normalization_params: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
```

**Process per question**:
1. Retrieve top-k chunks and normalize scores
2. Compute confidence and apply Level 1 threshold
3. If filtered results are empty -> Level 1 abstention
4. If results pass -> generate answer -> check for `ABSTENTION:` prefix (Level 2)
5. Record outcome: abstained/answered, level, confidence, reason

**Output**: Report dict with `config`, `metrics`, and `results` keys

### Combined Pipeline (`run_and_report()`)

Convenience function that chains Level 1 sweep -> best threshold -> full evaluation:

1. Sweep Level 1 thresholds (no LLM)
2. Find threshold that maximises F0.5
3. Run full two-level evaluation at that threshold

## Threshold Optimisation

The threshold is optimised by maximising **F0.5** on the evaluation
dataset. F0.5 weights precision at twice the weight of recall:

```python
best = max(sweep, key=lambda s: s["metrics"]["f0_5"])
best_threshold = best["threshold"]
```

**Why F0.5 rather than F1?** The evaluation set is 25 % unanswerable
questions, and Level-1 abstention already reaches recall ≈ 1.0 at modest
thresholds. Further tightening — which F1 still rewards because each
additional correct abstention is weighted equally with precision loss —
pushes the threshold up to 0.95–1.0 and strips the system of answers on
answerable questions. F0.5 stops earlier, trading a marginal recall hit
for a meaningful gain in usable coverage.

The default threshold in the production pipeline is 0.3 (from
`models/constants.py: DEFAULT_THRESHOLD`). The evaluation sweep tests
whether a different threshold would perform better on the evaluation
data.

## CLI Entry Point

In the unified CLI (`__main__.py`), the abstention step:

1. Creates a `BM25Retriever` and `OllamaGenerator`
2. For each KB: indexes, runs Level 1 sweep, determines best threshold
3. Runs full evaluation at the best threshold
4. Saves results

```bash
# Run abstention evaluation
python -m evaluation --abstention

# With custom model
python -m evaluation --abstention --ollama-model mistral:latest
```

## Output File

| File | Content |
|---|---|
| `abstention-evaluation.json` | Per-KB results with Level 1 sweep and full evaluation metrics |

### Report Format

```json
{
  "config": {
    "k": 5,
    "threshold": 0.35,
    "normalization_strategy": "bm25",
    "normalization_params": {}
  },
  "metrics": {
    "precision": 0.85,
    "recall": 0.90,
    "f1": 0.87,
    "false_abstention_rate": 0.05,
    "coverage": 0.92,
    "num_correct_abstention": 18,
    "num_incorrect_abstention": 3,
    "num_missing_abstention": 2,
    "num_answered": 77,
    "num_total": 100,
    "threshold": 0.35
  },
  "results": [ "..." ],
  "level1_sweep": [ "..." ],
  "knowledge_base": "stpo"
}
```
