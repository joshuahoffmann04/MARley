# Evaluation

**Module:** `evaluation/`
**Metrics:** Precision@k, Recall@k, MRR
**Test files:** `evaluation/tests/test_metrics.py`, `evaluation/tests/test_evaluate.py`

The evaluation harness measures retrieval quality by comparing retrieved chunks against manually annotated ground-truth chunk IDs. It supports evaluating any `Retriever` implementation against any of the three knowledge bases.

---

## Evaluation Methodology

### Metrics (from Thesis Proposal)

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision@k** | \|relevant ∩ retrieved[:k]\| / k | Proportion of top-k results that are relevant |
| **Recall@k** | \|relevant ∩ retrieved[:k]\| / \|relevant\| | Proportion of all relevant chunks found in top-k |
| **MRR** | 1 / rank of first relevant result | How early the first relevant result appears |

All metrics are averaged over all evaluated queries to produce macro-averages.

### Query Handling

- **Unanswerable questions** (`expected_abstention: true`) are skipped during retrieval evaluation, as they have no relevant chunks by definition.
- **Questions with empty `relevant_chunks`** are also skipped (no ground truth available).

---

## Evaluation Data Files

Three annotated evaluation files, one per knowledge base:

| File | Knowledge Base | Annotated Questions |
|---|---|---|
| `data/testing/evaluation-stpo.json` | StPO chunks (142 text + table) | 75 |
| `data/testing/evaluation-faq-stpo.json` | FAQ-StPO chunks (999 Q/A) | 75 |
| `data/testing/evaluation-faq-ao.json` | FAQ-AO chunks (50 Q/A) | 21 |

Each file contains the same 100 questions from the master `evaluation.json`, with `relevant_chunks` populated for the respective knowledge base. Unanswerable questions (25) have empty `relevant_chunks` in all files.

### File Structure

```json
{
  "metadata": {
    "version": "1.0",
    "created": "2026-03-06",
    "description": "Evaluation dataset with relevant chunks for StPO chunks (PDF text + tables).",
    "knowledge_base": "stpo"
  },
  "questions": [
    {
      "id": "eval-001",
      "question": "How long is the standard study period for the master's program?",
      "reference_answer": "The standard study period (Regelstudienzeit) is 4 semesters.",
      "category": "direct",
      "relevant_chunks": ["par-7-txt-1"],
      "expected_abstention": false
    }
  ]
}
```

### Annotation Criteria

- Only chunks that **directly contain the answer** are marked as relevant.
- For multi-source questions, all required chunks are listed.
- Thematically related but non-answering chunks are excluded.

---

## Combining Knowledge Bases

The evaluation is designed to test each knowledge base separately. To evaluate a combined retrieval (e.g., StPO + FAQ-StPO):

1. Merge chunk corpora: `combined = stpo_chunks + faq_stpo_chunks`
2. Merge relevant_chunks from both evaluation files for each question
3. Run evaluation over the combined set

---

## Usage

### Programmatic

```python
from src.marley.retrieval import BM25Retriever, load_chunks
from evaluation.evaluate import run_and_report

# Setup
chunks = load_chunks("data/chunks/stpo-chunks.json")
retriever = BM25Retriever()
retriever.index(chunks)

# Evaluate
report = run_and_report(
    retriever,
    "data/testing/evaluation-stpo.json",
    k=5,
)
print(report["metrics"])
```

### Functions

| Function | Description |
|---|---|
| `load_evaluation(path)` | Load annotated evaluation JSON. |
| `run_evaluation(retriever, questions, k)` | Run retrieval and compute metrics. |
| `run_and_report(retriever, path, k)` | Full pipeline: load, run, report. |

---

## Baseline Results

### BM25 (Okapi BM25, lowercase whitespace tokenization)

| Knowledge Base | P@1 | R@1 | MRR | P@5 | R@5 | MRR | Queries |
|---|---|---|---|---|---|---|---|
| StPO (142 chunks) | 0.320 | 0.238 | 0.320 | 0.117 | 0.438 | 0.400 | 75 |
| FAQ-StPO (999 chunks) | 0.227 | 0.153 | 0.227 | 0.115 | 0.394 | 0.332 | 75 |
| FAQ-AO (50 chunks) | 0.571 | 0.548 | 0.571 | 0.181 | 0.857 | 0.660 | 21 |

### Vector (all-mpnet-base-v2, cosine similarity)

| Knowledge Base | P@1 | R@1 | MRR | P@5 | R@5 | MRR | Queries |
|---|---|---|---|---|---|---|---|
| StPO (142 chunks) | 0.400 | 0.286 | 0.400 | 0.157 | 0.512 | 0.477 | 75 |
| FAQ-StPO (999 chunks) | 0.440 | 0.321 | 0.440 | 0.173 | 0.576 | 0.532 | 75 |
| FAQ-AO (50 chunks) | 0.762 | 0.738 | 0.762 | 0.210 | 1.000 | 0.861 | 21 |

### Comparison

Vector retrieval outperforms BM25 across all knowledge bases and all metrics:

- **StPO:** R@5 improves from 0.438 to 0.512 (+17%), MRR from 0.400 to 0.477 (+19%).
- **FAQ-StPO:** R@5 improves from 0.394 to 0.576 (+46%), MRR from 0.332 to 0.532 (+60%). This is the largest gain, since FAQ entries share many keywords but differ semantically — exactly where BM25 struggles.
- **FAQ-AO:** R@5 reaches 1.000 (perfect recall), MRR jumps from 0.660 to 0.861 (+30%). The small corpus and direct questions make this the easiest task.

The gains are especially pronounced for FAQ-StPO, where BM25's keyword matching is confused by the many similarly-worded FAQ entries. Dense embeddings capture semantic similarity more effectively in this setting.

---

## Module Structure

```
evaluation/
├── __init__.py
├── metrics.py          # Precision@k, Recall@k, MRR, evaluate_retriever()
├── evaluate.py         # Runner: load_evaluation(), run_evaluation(), run_and_report()
└── tests/
    ├── __init__.py
    ├── test_metrics.py  # 21 unit tests for all metric functions
    └── test_evaluate.py # 13 unit tests for the evaluation runner
```
