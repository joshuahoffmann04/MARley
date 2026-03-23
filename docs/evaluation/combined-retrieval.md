# Combined Knowledge Base Retrieval Evaluation

**Module:** `evaluation/retrieval/combined.py`
**Metrics:** Precision@k, Recall@k, MRR
**Test file:** `evaluation/tests/retrieval/test_combined.py`

This evaluation measures whether combining multiple knowledge bases improves retrieval quality compared to single-KB retrieval. Two combination strategies are compared: merged pool and separate retrieval with fusion.

---

## Objective

The thesis proposal (Goal G2) states: "Chunks from all three knowledge sources will compete on equal terms in the retrieval ranking." This evaluation tests this claim by measuring retrieval quality over combined KBs and comparing against the established single-KB baselines.

---

## Combination Strategies

### Strategy 1: Merged Pool

All chunks from the selected KBs are combined into a single corpus. One retrieval index is built over the merged corpus, and retrieval operates exactly as in the single-KB case.

```
KB_1 chunks + KB_2 chunks + ... -> single index -> retrieve(query, k)
```

No new retriever logic is needed. The existing retrievers (BM25, Vector, Hybrid) work unchanged on the larger corpus.

### Strategy 2: Separate Retrieval + Fusion

Each KB gets its own retriever instance indexed on its own chunks. For each query, retrieval runs independently on each KB, and the per-KB result lists are fused using Reciprocal Rank Fusion (RRF).

```
KB_1 -> retriever_1.retrieve(query) -+
KB_2 -> retriever_2.retrieve(query) -+-> rrf_fuse() -> fused top-k
KB_3 -> retriever_3.retrieve(query) -+
```

RRF is rank-based (not score-based), which is critical because retrieval scores from different indices are not directly comparable (e.g., BM25 IDF values depend on corpus statistics).

**RRF formula:** `score(d) = sum( weight_i / (k_rrf + rank_i(d)) )` with `k_rrf = 60` and uniform weights (all 1.0) by default.

### Why both strategies?

The merged pool is the simpler, more natural approach — all chunks compete equally. The fusion strategy tests whether preserving per-KB retrieval context and merging at the result level outperforms a flat merge. Comparing both provides evidence for the best approach.

---

## KB Combinations

| ID | Combination | Chunks | Rationale |
|---|---|---|---|
| stpo+faq-stpo | StPO + FAQ-StPO | 153 + 1039 = 1192 | Same regulation, different formats |
| stpo+faq-ao | StPO + FAQ-AO | 153 + 0 = 153 | Formal text + real student questions |
| faq-stpo+faq-ao | FAQ-StPO + FAQ-AO | 1039 + 0 = 1039 | Two FAQ sources |
| all | StPO + FAQ-StPO + FAQ-AO | 153 + 1039 + 0 = 1192 | All three combined |

---

## Evaluation Data

For each question, relevant chunks from all included KBs are merged using set union:

```
question eval-001:
  evaluation-stpo.json:      relevant_chunks = ["par-7-txt-1"]
  evaluation-faq-stpo.json:  relevant_chunks = ["faq-stpo-stpo-0012"]
  merged:                    relevant_chunks = ["par-7-txt-1", "faq-stpo-stpo-0012"]
```

A question is evaluable if it has at least one relevant chunk across any included KB. Unanswerable questions (`expected_abstention: true`) are skipped. This yields **75 evaluable queries** for all combinations.

---

## Results

All configurations evaluated with 75 queries at k=1 and k=5.

### BM25

#### Merged Pool

| Combination | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.253 | 0.102 | 0.253 | 0.125 | 0.225 | 0.347 | 75 |
| stpo+faq-ao | 0.240 | 0.167 | 0.240 | 0.131 | 0.363 | 0.364 | 75 |
| faq-stpo+faq-ao | 0.253 | 0.175 | 0.253 | 0.133 | 0.371 | 0.366 | 75 |
| all | 0.267 | 0.103 | 0.267 | 0.149 | 0.232 | 0.386 | 75 |

#### Fusion (RRF, k_rrf=60)

| Combination | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.267 | 0.117 | 0.267 | 0.168 | 0.294 | 0.380 | 75 |
| stpo+faq-ao | 0.240 | 0.167 | 0.240 | 0.125 | 0.356 | 0.358 | 75 |
| faq-stpo+faq-ao | 0.227 | 0.155 | 0.227 | 0.128 | 0.355 | 0.341 | 75 |
| all | 0.267 | 0.117 | 0.267 | 0.195 | 0.301 | 0.395 | 75 |

### Vector (all-mpnet-base-v2)

#### Merged Pool

| Combination | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.413 | 0.174 | 0.413 | 0.227 | 0.388 | 0.527 | 75 |
| stpo+faq-ao | 0.387 | 0.257 | 0.387 | 0.179 | 0.480 | 0.503 | 75 |
| faq-stpo+faq-ao | 0.453 | 0.316 | 0.453 | 0.197 | 0.571 | 0.569 | 75 |
| all | 0.440 | 0.183 | 0.440 | 0.245 | 0.386 | 0.560 | 75 |

#### Fusion (RRF, k_rrf=60)

| Combination | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.413 | 0.177 | 0.413 | 0.251 | 0.428 | 0.527 | 75 |
| stpo+faq-ao | 0.373 | 0.240 | 0.373 | 0.181 | 0.472 | 0.472 | 75 |
| faq-stpo+faq-ao | 0.413 | 0.293 | 0.413 | 0.200 | 0.556 | 0.545 | 75 |
| all | 0.413 | 0.177 | 0.413 | 0.272 | 0.419 | 0.535 | 75 |

### Hybrid (BM25 + Vector, RRF with k_rrf=60)

#### Merged Pool

| Combination | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.293 | 0.120 | 0.293 | 0.187 | 0.335 | 0.449 | 75 |
| stpo+faq-ao | 0.320 | 0.214 | 0.320 | 0.173 | 0.479 | 0.455 | 75 |
| faq-stpo+faq-ao | 0.333 | 0.231 | 0.333 | 0.187 | 0.536 | 0.482 | 75 |
| all | 0.320 | 0.128 | 0.320 | 0.216 | 0.348 | 0.469 | 75 |

#### Fusion (RRF, k_rrf=60)

| Combination | P@1 | R@1 | MRR@1 | P@5 | R@5 | MRR@5 | Queries |
|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.347 | 0.158 | 0.347 | 0.245 | 0.422 | 0.473 | 75 |
| stpo+faq-ao | 0.307 | 0.211 | 0.307 | 0.181 | 0.482 | 0.429 | 75 |
| faq-stpo+faq-ao | 0.347 | 0.237 | 0.347 | 0.176 | 0.490 | 0.474 | 75 |
| all | 0.347 | 0.150 | 0.347 | 0.253 | 0.390 | 0.467 | 75 |

---

## Comparison with Single-KB Baselines

### Single-KB Baselines (from [retrieval.md](retrieval.md))

| Retriever | KB | P@5 | R@5 | MRR@5 |
|---|---|---|---|---|
| BM25 | StPO | 0.112 | 0.420 | 0.347 |
| BM25 | FAQ-StPO | 0.119 | 0.407 | 0.357 |
| BM25 | FAQ-AO | — | — | — |
| Vector | StPO | 0.147 | 0.488 | 0.464 |
| Vector | FAQ-StPO | 0.173 | 0.563 | 0.530 |
| Vector | FAQ-AO | — | — | — |
| Hybrid | StPO | 0.157 | 0.547 | 0.432 |
| Hybrid | FAQ-StPO | 0.162 | 0.536 | 0.476 |
| Hybrid | FAQ-AO | — | — | — |

**Note:** Direct comparison requires care. Single-KB baselines evaluate only on questions with relevant chunks in that specific KB (75 for StPO/FAQ-StPO, 21 for FAQ-AO). Combined-KB results evaluate all 75 non-unanswerable questions with merged relevant_chunks. The query sets overlap but are not identical, as a question may gain relevant chunks from a newly included KB.

### Best Combined Results vs. Best Single-KB

| Retriever | Best Combined (R@5) | Config | Best Single-KB (R@5) | KB |
|---|---|---|---|---|
| BM25 | 0.371 | faq-stpo+faq-ao merged | 0.420 | StPO |
| Vector | 0.571 | faq-stpo+faq-ao merged | 0.563 | FAQ-StPO |
| Hybrid | 0.536 | faq-stpo+faq-ao merged | 0.547 | StPO |

**Note:** Combined-KB baseline results above were computed before the FAQ-AO content reset (0 chunks). Combinations involving FAQ-AO effectively operate on only the non-empty KBs. These combined results will be refreshed when FAQ-AO content is available.

**Note:** FAQ-AO currently has 0 chunks (placeholder). Combined-KB results involving FAQ-AO reflect only the non-empty KBs. These baselines will be updated when FAQ-AO content is available.

---

## Analysis

### Strategy Comparison: Merged Pool vs. Fusion

**BM25:**
- Fusion outperforms merged pool for stpo+faq-stpo (R@5: 0.294 vs 0.225, +31%) and for the all combination (R@5: 0.301 vs 0.232, +30%).
- For stpo+faq-ao and faq-stpo+faq-ao, merged pool has a slight edge.
- Overall, BM25 fusion benefits from preserving per-corpus IDF statistics.

**Vector:**
- Results are close between strategies. Fusion slightly improves stpo+faq-stpo (R@5: 0.428 vs 0.388, +10%) and all (R@5: 0.419 vs 0.386, +9%).
- faq-stpo+faq-ao slightly favors merged pool (R@5: 0.571 vs 0.556).
- Dense embeddings are less affected by corpus composition changes.

**Hybrid:**
- Fusion consistently matches or improves over merged pool for stpo+faq-stpo (R@5: 0.422 vs 0.335, +26%) and all (R@5: 0.390 vs 0.348, +12%).
- faq-stpo+faq-ao slightly favors merged pool (R@5: 0.536 vs 0.490).

**Key insight:** Fusion tends to outperform merged pool when combining KBs of very different sizes (stpo+faq-stpo: 153 vs 1039 chunks), because per-KB retrieval preserves the smaller KB's results from being drowned out by the larger one.

### Combination Comparison

**faq-stpo+faq-ao** consistently achieves the highest R@5 across all retrievers in the merged pool strategy (0.371, 0.571, 0.536). This combination benefits from having two complementary FAQ sources.

**stpo+faq-ao** performs well across strategies, benefiting from the small FAQ-AO corpus that provides high-precision matches for direct student questions.

**all (three KBs combined)** does not consistently outperform the best two-KB combinations. The large FAQ-StPO corpus (1039 chunks) can dilute results when merged with the smaller KBs.

### Retriever Ranking

Across all 24 configurations, the retriever ranking is consistent with single-KB findings:

1. **Vector** — best overall (highest P@5, R@5, MRR@5 in most configurations)
2. **Hybrid** — second, with strong recall
3. **BM25** — lowest metrics across all configurations

---

## Module Structure

```
evaluation/retrieval/
├── combined.py             # Combined-KB evaluation (this module)
├── evaluate.py             # Single-KB evaluation runner
└── metrics.py              # Shared metrics (Precision@k, Recall@k, MRR)

src/marley/retrieval/
└── fusion.py               # RRF fusion utility (shared by HybridRetriever + combined eval)
```

### Functions

| Function | Description |
|---|---|
| `merge_chunks(*paths)` | Load and concatenate chunks from multiple JSON files. Validates no duplicate chunk_ids. |
| `merge_evaluation_data(eval_paths)` | Merge evaluation datasets from multiple KBs. Uses set union for relevant_chunks. |
| `run_merged_pool_evaluation(retriever, chunk_paths, eval_paths, k)` | Strategy 1: merged corpus, single index. |
| `run_fusion_evaluation(retriever_factory, chunk_paths, eval_paths, k, k_rrf)` | Strategy 2: per-KB retrievers, RRF fusion. |

---

## Usage

### Merged Pool

```python
from src.marley.retrieval import BM25Retriever
from evaluation.retrieval.combined import run_merged_pool_evaluation

report = run_merged_pool_evaluation(
    retriever=BM25Retriever(),
    chunk_paths={
        "stpo": "data/chunks/stpo-chunks.json",
        "faq-stpo": "data/chunks/faq-stpo-chunks.json",
    },
    eval_paths={
        "stpo": "data/testing/evaluation-stpo.json",
        "faq-stpo": "data/testing/evaluation-faq-stpo.json",
    },
    k=5,
)
print(report["metrics"])
```

### Separate Retrieval + Fusion

```python
from src.marley.retrieval import BM25Retriever
from evaluation.retrieval.combined import run_fusion_evaluation

report = run_fusion_evaluation(
    retriever_factory=BM25Retriever,
    chunk_paths={
        "stpo": "data/chunks/stpo-chunks.json",
        "faq-stpo": "data/chunks/faq-stpo-chunks.json",
    },
    eval_paths={
        "stpo": "data/testing/evaluation-stpo.json",
        "faq-stpo": "data/testing/evaluation-faq-stpo.json",
    },
    k=5,
    k_rrf=60,
)
print(report["metrics"])
```
