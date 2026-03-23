# Manual Evaluation Framework

**Module:** `evaluation/manual/`
**Purpose:** Human correctness assessment for generated answers
**Test files:** `evaluation/tests/manual/test_models.py`, `evaluation/tests/manual/test_prepare.py`, `evaluation/tests/manual/test_metrics.py`

The manual evaluation framework replaces the LLM-as-Judge with a human evaluation web UI for answer correctness assessment. Using the same LLM for both generation and judgement introduces potential bias; manual evaluation provides more reliable and defensible correctness assessments for the thesis.

**See also:** [Evaluation Overview](overview.md) | [Generation Evaluation](generation.md)

---

## Methodology

### Overview

1. The generation evaluation runner produces answers for each (question, distractor_level) pair.
2. These results are converted to **evaluation items** — a generic format for human assessment.
3. A **web UI** (FastAPI + Jinja2) presents each item to the evaluator.
4. The evaluator assigns a **judgement** from a 6-value schema.
5. Judgements are saved incrementally — sessions can be paused and resumed.
6. **Metrics** are computed by joining items with their judgements.

### Judgement Schema

| Value | When to use | Applies to |
|---|---|---|
| `correct` | Answer is semantically correct and complete | Answerable questions |
| `partially_correct` | Answer has some correct info but is incomplete or has minor issues | Answerable questions |
| `incorrect` | Answer is factually wrong, contains hallucinations, or misses core info | Answerable questions |
| `correct_abstention` | System correctly refused to answer an unanswerable question | Unanswerable questions |
| `incorrect_abstention` | System refused to answer but should have been able to (false negative) | Answerable questions |
| `missing_abstention` | System answered but should have abstained (false positive) | Unanswerable questions |

### Decision Tree

```
Is the question answerable?
+-- YES
|   +-- Did the system provide an answer?
|   |   +-- YES
|   |   |   +-- Answer is fully correct          -> correct
|   |   |   +-- Answer is partially correct      -> partially_correct
|   |   |   +-- Answer is incorrect              -> incorrect
|   |   +-- NO (system abstained)
|   |       +-- Should have answered              -> incorrect_abstention
|
+-- NO (unanswerable)
    +-- Did the system abstain?
    |   +-- YES -> correct_abstention
    |   +-- NO (system answered anyway)
    |       +-- Should have abstained             -> missing_abstention
```

---

## Metrics

### Strict Accuracy

`correct / total_judged` — only fully correct answers count.

### Lenient Accuracy

`(correct + partially_correct) / total_judged` — partial credit included.

### Abstention Precision

`correct_abstention / (correct_abstention + incorrect_abstention)` — of all abstentions, how many were justified?

### Abstention Recall

`correct_abstention / (correct_abstention + missing_abstention)` — of all unanswerable questions, how many did the system abstain on?

Both strict and lenient accuracy are reported overall and grouped by distractor level.

---

## Data Model

### EvaluationItem

```python
@dataclass
class EvaluationItem:
    id: str                     # e.g. "gen-stpo-eval-001-d0"
    question: str               # The question text
    generated_answer: str       # The system's answer
    reference_answer: str       # The ground-truth answer
    category: str               # direct, multi-source, unanswerable
    expected_abstention: bool   # True for unanswerable questions
    metadata: dict              # Evaluation-type-specific metadata
```

**Item ID format:** `{type}-{kb}-{question_id}-d{num_distractors}`

**Metadata fields (generation evaluation):**
- `question_id`: Original question ID
- `knowledge_base`: KB identifier
- `num_distractors`: Number of distractor chunks
- `context_chunk_ids`: Chunk IDs in the generation context
- `evaluation_type`: `"generation"`
- `generator_model`: Model used for generation

### ManualJudgement

```python
@dataclass
class ManualJudgement:
    item_id: str                # References EvaluationItem.id
    judgement: Judgement         # One of the 6 judgement values
    notes: str                  # Optional free-text notes
    timestamp: str              # ISO 8601 datetime
```

### File Formats

**Items** (`data/testing/manual-eval-items-{source}.json`): Immutable, generated once from evaluation results.

**Judgements** (`data/testing/manual-judgements-{source}.json`): Append-only, incrementally saved. If an item is judged multiple times, the latest entry wins.

---

## Usage

### One-Time Setup

Prepare evaluation items from existing generation results:

```python
from evaluation.manual.prepare import prepare_generation_items
from evaluation.manual.models import save_items

for kb in ["stpo", "faq-stpo", "faq-ao"]:
    items = prepare_generation_items(
        f"data/testing/generation-eval-{kb}.json",
        knowledge_base=kb,
        eval_dataset_path=f"data/testing/evaluation-{kb}.json",
    )
    save_items(items, f"data/testing/manual-eval-items-generation-{kb}.json", metadata={
        "source": f"generation-{kb}",
        "evaluation_type": "generation",
        "total_items": len(items),
    })
    print(f"{kb}: {len(items)} items prepared")
```

### Evaluation Session

```bash
# Start the evaluation UI
python -m evaluation.manual --items-dir data/testing/ --port 8000

# Open browser at http://localhost:8000
# Select source, apply filters, start judging
# Close browser anytime — progress is saved incrementally
# Resume later — previously judged items are marked
```

### Compute Metrics

```python
from evaluation.manual.models import load_items, load_judgements
from evaluation.manual.metrics import compute_manual_metrics

items = load_items("data/testing/manual-eval-items-generation-stpo.json")
judgements = load_judgements("data/testing/manual-judgements-generation-stpo.json")
metrics = compute_manual_metrics(items, judgements, knowledge_base="stpo")
print(f"Strict accuracy: {metrics.strict_accuracy:.3f}")
print(f"Lenient accuracy: {metrics.lenient_accuracy:.3f}")
```

---

## UI Guide

### Layout

The evaluation UI displays one item at a time with:

- **Progress bar** — judged/total count, updates live
- **Filter bar** — filter by source, distractor count, category, status (pending/judged/all)
- **Question** — prominently displayed
- **Side-by-side answers** — generated answer left, reference answer right
- **Metadata** — KB, distractor count, category, expected behaviour
- **Notes field** — optional free text for edge cases
- **Judgement buttons** — two rows:
  - Row 1 (answer): Correct, Partially Correct, Incorrect
  - Row 2 (abstention): Correct Abstention, Incorrect Abstention, Missing Abstention
  - The appropriate row is highlighted based on `expected_abstention`
- **Navigation** — Previous/Next buttons

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` | Correct |
| `2` | Partially Correct |
| `3` | Incorrect |
| `4` | Correct Abstention |
| `5` | Incorrect Abstention |
| `6` | Missing Abstention |
| `Left Arrow` | Previous item |
| `Right Arrow` | Next item |

After clicking a judgement, the UI auto-advances to the next unjudged item.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET /` | | Serve the evaluation UI |
| `GET /api/sources` | | List available evaluation item sources |
| `GET /api/items` | `?source=...&filter_kb=...&filter_distractors=...&filter_status=...&filter_category=...` | Return items with filters |
| `GET /api/items/{item_id}` | `?source=...` | Return a single item |
| `GET /api/progress` | `?source=...` | Progress stats |
| `POST /api/judgements` | `?source=...` Body: `{item_id, judgement, notes}` | Save a judgement |
| `GET /api/judgements` | `?source=...` | All judgements for a source |

---

## Module Structure

```
evaluation/
├── manual/
│   ├── __init__.py
│   ├── models.py          # EvaluationItem, ManualJudgement, Judgement enum, I/O
│   ├── prepare.py         # Convert generation results -> evaluation items
│   ├── metrics.py         # Compute metrics from manual judgements
│   ├── app.py             # FastAPI application
│   ├── __main__.py        # CLI entry point (python -m evaluation.manual)
│   ├── static/
│   │   ├── style.css      # UI styling
│   │   └── app.js         # Frontend logic
│   └── templates/
│       └── evaluate.html  # Jinja2 template
└── tests/
    └── manual/
        ├── __init__.py
        ├── test_models.py    # Data model validation tests
        ├── test_prepare.py   # Item preparation tests
        └── test_metrics.py   # Manual metric computation tests
```
