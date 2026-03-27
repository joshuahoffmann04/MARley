# Retrieval Evaluation

**Module:** `evaluation/retrieval/`
**Metrics:** Precision@k, Recall@k, MRR, MAP, F1@k, Jaccard@k
**Test files:** `evaluation/tests/retrieval/test_metrics.py`, `evaluation/tests/retrieval/test_evaluate.py`

The evaluation harness measures retrieval quality by comparing retrieved chunks against manually annotated ground-truth chunk IDs. It supports evaluating any `Retriever` implementation against any of the three knowledge bases.

**See also:** [Evaluation Overview](overview.md) | [Combined-KB Retrieval Evaluation](combined-retrieval.md)

---

## Evaluation Methodology

### Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision@k** | \|relevant ∩ retrieved[:k]\| / k | Proportion of top-k results that are relevant |
| **Recall@k** | \|relevant ∩ retrieved[:k]\| / \|relevant\| | Proportion of all relevant chunks found in top-k |
| **F1@k** | 2 · P@k · R@k / (P@k + R@k) | Harmonic mean of precision and recall at k |
| **MRR** | 1 / rank of first relevant result | How early the first relevant result appears |
| **MAP** | mean(AP@k) where AP = (1/\|rel\|) Σ P@i · rel(i) | Mean rank quality across all relevant hits |
| **Jaccard@k** | \|relevant ∩ retrieved[:k]\| / \|relevant ∪ retrieved[:k]\| | Set overlap between retrieved and relevant |

All metrics are averaged over all evaluated queries to produce macro-averages. MRR is effectively bounded by k because the retriever returns at most k results.

**MAP vs. MRR:** MRR only measures the position of the *first* relevant hit. MAP evaluates the precision at *every* relevant hit position, rewarding retrievers that rank all relevant documents highly — not just the first one. This is especially important for multi-source questions where multiple chunks are needed to construct a complete answer.

**Jaccard@k** provides a set-level view: unlike Precision (normalized by k) and Recall (normalized by |relevant|), Jaccard normalizes by the union, penalizing both irrelevant retrievals and missed relevant documents simultaneously.

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

All results evaluated over 75 queries at k=1 and k=5. FAQ-AO has 0 chunks (placeholder) and is omitted.

### BM25 (Okapi BM25, lowercase whitespace tokenization)

| Knowledge Base | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| StPO (153 chunks) | 0.280 | 0.207 | 0.229 | 0.280 | 0.207 | 0.207 | 0.115 | 0.413 | 0.175 | 0.360 | 0.300 | 0.109 |
| FAQ-StPO (1039 chunks) | 0.280 | 0.191 | 0.220 | 0.280 | 0.191 | 0.191 | 0.131 | 0.402 | 0.190 | 0.383 | 0.280 | 0.116 |

### Vector (all-mpnet-base-v2, cosine similarity)

| Knowledge Base | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| StPO (153 chunks) | 0.400 | 0.281 | 0.319 | 0.400 | 0.281 | 0.281 | 0.152 | 0.494 | 0.225 | 0.474 | 0.388 | 0.144 |
| FAQ-StPO (1039 chunks) | 0.453 | 0.319 | 0.361 | 0.453 | 0.319 | 0.319 | 0.184 | 0.557 | 0.267 | 0.557 | 0.444 | 0.170 |

### Hybrid (BM25 + Vector, RRF with k_rrf=60)

| Knowledge Base | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| StPO (153 chunks) | 0.280 | 0.207 | 0.229 | 0.280 | 0.207 | 0.207 | 0.160 | 0.540 | 0.239 | 0.446 | 0.373 | 0.151 |
| FAQ-StPO (1039 chunks) | 0.280 | 0.191 | 0.220 | 0.280 | 0.191 | 0.191 | 0.168 | 0.533 | 0.248 | 0.497 | 0.387 | 0.156 |

### Comparison

#### BM25 vs. Vector

Vector retrieval outperforms BM25 across all knowledge bases and all metrics:

- **StPO:** R@5 improves from 0.413 to 0.494 (+20%), MAP@5 from 0.300 to 0.388 (+29%), F1@5 from 0.175 to 0.225 (+29%).
- **FAQ-StPO:** R@5 improves from 0.402 to 0.557 (+39%), MAP@5 from 0.280 to 0.444 (+59%), F1@5 from 0.190 to 0.267 (+41%). This is the largest gain, since FAQ entries share many keywords but differ semantically — exactly where BM25 struggles.

The gains are especially pronounced for FAQ-StPO, where BM25's keyword matching is confused by the many similarly-worded FAQ entries. Dense embeddings capture semantic similarity more effectively in this setting.

#### Hybrid (RRF) vs. Individual Retrievers

Hybrid retrieval (RRF) shows mixed results compared to the individual strategies:

- **StPO:** R@5 improves to 0.540, the **best recall across all strategies** (+9% over Vector, +31% over BM25). MAP@5 of 0.373 sits between BM25 (0.300) and Vector (0.388). P@1 and MRR@1 match BM25 (0.280) rather than Vector (0.400).
- **FAQ-StPO:** R@5 of 0.533 is better than BM25 (0.402, +33%) but slightly below Vector (0.557, −4%). MAP@5 of 0.387 improves over BM25 (0.280, +38%) but falls below Vector (0.444, −13%). The k@1 metrics match BM25, not Vector.

**Key insight:** RRF's P@1 and MRR@1 match BM25 in all cases. This occurs because both retrievers contribute equally to the RRF score at rank 1, and BM25 results are processed first in tie-breaking.

**MAP analysis:** MAP reveals that Hybrid retrieval ranks all relevant documents more consistently than BM25 (MAP@5: 0.373/0.387 vs. 0.300/0.280), but still falls below Vector (0.388/0.444). This aligns with the recall findings — Hybrid retrieves more relevant documents but does not place them as highly as Vector.

**Practical implication:** RRF excels at recall — it finds more relevant documents by combining both retriever pools. For precision-sensitive use cases (where only the top-1 or top-2 results matter), pure Vector retrieval remains the better choice. For the downstream generation stage, the higher recall of hybrid retrieval may be more valuable, as the language model can select the most relevant information from a richer context.

---

## Module Structure

```
evaluation/
├── __init__.py
├── retrieval/
│   ├── __init__.py
│   ├── metrics.py          # P@k, R@k, F1@k, MRR, MAP, Jaccard@k, evaluate_retriever()
│   └── evaluate.py         # Runner: load_evaluation(), run_evaluation(), run_and_report()
└── tests/
    ├── __init__.py
    └── retrieval/
        ├── __init__.py
        ├── test_metrics.py  # 41 unit tests for all metric functions
        └── test_evaluate.py # 13 tests for the evaluation runner
```
