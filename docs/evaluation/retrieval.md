# Retrieval Evaluation

**Module:** `evaluation/retrieval/`
**Metrics:** Precision@k, Recall@k, MRR
**Test files:** `evaluation/tests/retrieval/test_metrics.py`, `evaluation/tests/retrieval/test_evaluate.py`

The evaluation harness measures retrieval quality by comparing retrieved chunks against manually annotated ground-truth chunk IDs. It supports evaluating any `Retriever` implementation against any of the three knowledge bases.

**See also:** [Evaluation Overview](overview.md) | [Combined-KB Retrieval Evaluation](combined-retrieval.md)

---

## Evaluation Methodology

### Metrics (from Thesis Proposal)

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision@k** | \|relevant ∩ retrieved[:k]\| / k | Proportion of top-k results that are relevant |
| **Recall@k** | \|relevant ∩ retrieved[:k]\| / \|relevant\| | Proportion of all relevant chunks found in top-k |
| **MRR** | 1 / rank of first relevant result | How early the first relevant result appears |

All metrics are averaged over all evaluated queries to produce macro-averages. MRR is effectively bounded by k because the retriever returns at most k results.

### Query Handling

- **Unanswerable questions** (`expected_abstention: true`) are skipped during retrieval evaluation, as they have no relevant chunks by definition.
- **Questions with empty `relevant_chunks`** are also skipped (no ground truth available).

---

## Evaluation Data Files

Three annotated evaluation files, one per knowledge base:

| File | Knowledge Base | Annotated Questions |
|---|---|---|
| `data/testing/evaluation-stpo.json` | StPO chunks (153 total) | 75 |
| `data/testing/evaluation-faq-stpo.json` | FAQ-StPO chunks (1039 Q/A) | 75 |
| `data/testing/evaluation-faq-ao.json` | FAQ-AO chunks (0 — placeholder) | 21 |

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
from evaluation.retrieval.evaluate import run_and_report

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

| Knowledge Base | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| StPO (153 chunks) | 0.267 | 0.200 | 0.267 | 0.112 | 0.420 | 0.347 | 75 |
| FAQ-StPO (1039 chunks) | 0.257 | 0.176 | 0.257 | 0.119 | 0.407 | 0.357 | 74 |
| FAQ-AO (0 chunks) | — | — | — | — | — | — | 0 |

### Vector (all-mpnet-base-v2, cosine similarity)

| Knowledge Base | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| StPO (153 chunks) | 0.400 | 0.294 | 0.400 | 0.147 | 0.488 | 0.464 | 75 |
| FAQ-StPO (1039 chunks) | 0.432 | 0.314 | 0.432 | 0.173 | 0.563 | 0.530 | 74 |
| FAQ-AO (0 chunks) | — | — | — | — | — | — | 0 |

### Hybrid (BM25 + Vector, RRF with k_rrf=60)

| Knowledge Base | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| StPO (153 chunks) | 0.267 | 0.200 | 0.267 | 0.157 | 0.547 | 0.432 | 75 |
| FAQ-StPO (1039 chunks) | 0.257 | 0.176 | 0.257 | 0.162 | 0.536 | 0.476 | 74 |
| FAQ-AO (0 chunks) | — | — | — | — | — | — | 0 |

### Comparison

#### BM25 vs. Vector

Vector retrieval outperforms BM25 across all knowledge bases and all metrics:

- **StPO:** R@5 improves from 0.420 to 0.488 (+16%), MRR@5 from 0.347 to 0.464 (+34%).
- **FAQ-StPO:** R@5 improves from 0.407 to 0.563 (+38%), MRR@5 from 0.357 to 0.530 (+48%). This is the largest gain, since FAQ entries share many keywords but differ semantically — exactly where BM25 struggles.
- **FAQ-AO:** Currently has 0 chunks (placeholder). Baselines will be established once FAQ-AO content is available.

The gains are especially pronounced for FAQ-StPO, where BM25's keyword matching is confused by the many similarly-worded FAQ entries. Dense embeddings capture semantic similarity more effectively in this setting.

#### Hybrid (RRF) vs. Individual Retrievers

Hybrid retrieval (RRF) shows mixed results compared to the individual strategies:

- **StPO:** R@5 improves to 0.547, the **best recall across all strategies** (+12% over Vector, +30% over BM25). However, P@1 and MRR@1 match BM25 (0.267) rather than Vector (0.400). MRR@5 of 0.432 sits between BM25 (0.347) and Vector (0.464).
- **FAQ-StPO:** R@5 of 0.536 is better than BM25 (0.407, +32%) but slightly below Vector (0.563, −5%). MRR@5 of 0.476 improves over BM25 (0.357, +33%) but falls below Vector (0.530, −10%). The k@1 metrics match BM25, not Vector.
- **FAQ-AO:** Currently has 0 chunks. Hybrid baselines pending.

**Key insight:** RRF's P@1 and MRR@1 are identical to BM25 in all cases. This occurs because both retrievers contribute equally to the RRF score at rank 1 (score = 1/(k_rrf+1)), and when both agree on rank 1, BM25's candidate wins; when they disagree, the tie-breaking favours whichever document appears first in iteration order. Since BM25 results are processed first, its top-1 candidate dominates.

**Practical implication:** RRF excels at recall — it finds more relevant documents by combining both retriever pools. For the StPO knowledge base, this is the best strategy. However, for precision-sensitive use cases (where only the top-1 or top-2 results matter), pure Vector retrieval remains the better choice. For the downstream generation stage, the higher recall of hybrid retrieval may be more valuable, as the language model can select the most relevant information from a richer context.

---

## Module Structure

```
evaluation/
├── __init__.py
├── retrieval/
│   ├── __init__.py
│   ├── metrics.py          # Precision@k, Recall@k, MRR, evaluate_retriever()
│   └── evaluate.py         # Runner: load_evaluation(), run_evaluation(), run_and_report()
└── tests/
    ├── __init__.py
    └── retrieval/
        ├── __init__.py
        ├── test_metrics.py  # 21 unit tests for all metric functions
        └── test_evaluate.py # 13 tests for the evaluation runner
```
