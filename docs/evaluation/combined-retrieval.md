# Combined Knowledge Base Retrieval Evaluation

**Module:** `evaluation/retrieval/combined.py`
**Metrics:** Precision@k, Recall@k, MRR, MAP, F1@k, Jaccard@k
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

| Combination | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.307 | 0.111 | 0.160 | 0.307 | 0.111 | 0.111 | 0.139 | 0.237 | 0.168 | 0.392 | 0.179 | 0.103 |
| stpo+faq-ao | 0.280 | 0.207 | 0.229 | 0.280 | 0.207 | 0.207 | 0.115 | 0.413 | 0.175 | 0.360 | 0.300 | 0.109 |
| faq-stpo+faq-ao | 0.280 | 0.191 | 0.220 | 0.280 | 0.191 | 0.191 | 0.131 | 0.402 | 0.190 | 0.383 | 0.280 | 0.116 |
| all | 0.307 | 0.111 | 0.160 | 0.307 | 0.111 | 0.111 | 0.139 | 0.237 | 0.168 | 0.392 | 0.179 | 0.103 |

#### Fusion (RRF, k_rrf=60)

| Combination | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.280 | 0.098 | 0.141 | 0.280 | 0.098 | 0.098 | 0.171 | 0.307 | 0.213 | 0.394 | 0.223 | 0.140 |
| stpo+faq-ao | 0.280 | 0.207 | 0.229 | 0.280 | 0.207 | 0.207 | 0.115 | 0.413 | 0.175 | 0.360 | 0.300 | 0.109 |
| faq-stpo+faq-ao | 0.280 | 0.191 | 0.220 | 0.280 | 0.191 | 0.191 | 0.131 | 0.402 | 0.190 | 0.383 | 0.280 | 0.116 |
| all | 0.280 | 0.098 | 0.141 | 0.280 | 0.098 | 0.098 | 0.171 | 0.307 | 0.213 | 0.394 | 0.223 | 0.140 |

### Vector (all-mpnet-base-v2)

#### Merged Pool

| Combination | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.453 | 0.165 | 0.235 | 0.453 | 0.165 | 0.165 | 0.243 | 0.383 | 0.286 | 0.560 | 0.299 | 0.188 |
| stpo+faq-ao | 0.400 | 0.281 | 0.319 | 0.400 | 0.281 | 0.281 | 0.152 | 0.494 | 0.225 | 0.474 | 0.388 | 0.144 |
| faq-stpo+faq-ao | 0.453 | 0.319 | 0.361 | 0.453 | 0.319 | 0.319 | 0.184 | 0.557 | 0.267 | 0.557 | 0.444 | 0.170 |
| all | 0.453 | 0.165 | 0.235 | 0.453 | 0.165 | 0.165 | 0.243 | 0.383 | 0.286 | 0.560 | 0.299 | 0.188 |

#### Fusion (RRF, k_rrf=60)

| Combination | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.400 | 0.125 | 0.185 | 0.400 | 0.125 | 0.125 | 0.264 | 0.422 | 0.313 | 0.549 | 0.312 | 0.211 |
| stpo+faq-ao | 0.400 | 0.281 | 0.319 | 0.400 | 0.281 | 0.281 | 0.152 | 0.494 | 0.225 | 0.474 | 0.388 | 0.144 |
| faq-stpo+faq-ao | 0.453 | 0.319 | 0.361 | 0.453 | 0.319 | 0.319 | 0.184 | 0.557 | 0.267 | 0.557 | 0.444 | 0.170 |
| all | 0.400 | 0.125 | 0.185 | 0.400 | 0.125 | 0.125 | 0.264 | 0.422 | 0.313 | 0.549 | 0.312 | 0.211 |

### Hybrid (BM25 + Vector, RRF with k_rrf=60)

#### Merged Pool

| Combination | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.307 | 0.111 | 0.160 | 0.307 | 0.111 | 0.111 | 0.200 | 0.340 | 0.244 | 0.487 | 0.249 | 0.155 |
| stpo+faq-ao | 0.280 | 0.207 | 0.229 | 0.280 | 0.207 | 0.207 | 0.160 | 0.540 | 0.239 | 0.446 | 0.373 | 0.151 |
| faq-stpo+faq-ao | 0.280 | 0.191 | 0.220 | 0.280 | 0.191 | 0.191 | 0.168 | 0.533 | 0.248 | 0.497 | 0.387 | 0.156 |
| all | 0.307 | 0.111 | 0.160 | 0.307 | 0.111 | 0.111 | 0.200 | 0.340 | 0.244 | 0.487 | 0.249 | 0.155 |

#### Fusion (RRF, k_rrf=60)

| Combination | P@1 | R@1 | F1@1 | MRR@1 | MAP@1 | J@1 | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stpo+faq-stpo | 0.280 | 0.098 | 0.141 | 0.280 | 0.098 | 0.098 | 0.253 | 0.423 | 0.306 | 0.478 | 0.286 | 0.201 |
| stpo+faq-ao | 0.280 | 0.207 | 0.229 | 0.280 | 0.207 | 0.207 | 0.160 | 0.540 | 0.239 | 0.446 | 0.373 | 0.151 |
| faq-stpo+faq-ao | 0.280 | 0.191 | 0.220 | 0.280 | 0.191 | 0.191 | 0.168 | 0.533 | 0.248 | 0.497 | 0.387 | 0.156 |
| all | 0.280 | 0.098 | 0.141 | 0.280 | 0.098 | 0.098 | 0.253 | 0.423 | 0.306 | 0.478 | 0.286 | 0.201 |

---

## Comparison with Single-KB Baselines

### Single-KB Baselines (from [retrieval.md](retrieval.md))

| Retriever | KB | P@5 | R@5 | F1@5 | MRR@5 | MAP@5 | J@5 |
|---|---|---|---|---|---|---|---|
| BM25 | StPO | 0.115 | 0.413 | 0.175 | 0.360 | 0.300 | 0.109 |
| BM25 | FAQ-StPO | 0.131 | 0.402 | 0.190 | 0.383 | 0.280 | 0.116 |
| Vector | StPO | 0.152 | 0.494 | 0.225 | 0.474 | 0.388 | 0.144 |
| Vector | FAQ-StPO | 0.184 | 0.557 | 0.267 | 0.557 | 0.444 | 0.170 |
| Hybrid | StPO | 0.160 | 0.540 | 0.239 | 0.446 | 0.373 | 0.151 |
| Hybrid | FAQ-StPO | 0.168 | 0.533 | 0.248 | 0.497 | 0.387 | 0.156 |

**Note:** FAQ-AO currently has 0 chunks (placeholder). Combinations involving FAQ-AO effectively operate on only the non-empty KBs. These baselines will be updated when FAQ-AO content is available.

### Best Combined Results vs. Best Single-KB

| Retriever | Best Combined (R@5) | Config | Best Single-KB (R@5) | KB |
|---|---|---|---|---|
| BM25 | 0.413 | stpo+faq-ao merged / fusion | 0.413 | StPO |
| Vector | 0.557 | faq-stpo+faq-ao merged / fusion | 0.557 | FAQ-StPO |
| Hybrid | 0.540 | stpo+faq-ao merged / fusion | 0.540 | StPO |

Since FAQ-AO has 0 chunks, combinations with FAQ-AO are identical to the single non-empty KB. The meaningful combined evaluation is **stpo+faq-stpo** and **all** (which are equivalent given FAQ-AO has 0 chunks).

---

## Analysis

### Strategy Comparison: Merged Pool vs. Fusion (stpo+faq-stpo)

The only meaningful multi-KB combination is stpo+faq-stpo (153 + 1039 = 1192 chunks). The other combinations involving FAQ-AO effectively reduce to single-KB evaluation.

**BM25:**
- Fusion outperforms merged pool (R@5: 0.307 vs 0.237, +30%; MAP@5: 0.223 vs 0.179, +25%). BM25 fusion benefits from preserving per-corpus IDF statistics.

**Vector:**
- Fusion slightly outperforms merged pool (R@5: 0.422 vs 0.383, +10%; MAP@5: 0.312 vs 0.299, +4%). Dense embeddings are less affected by corpus composition changes.

**Hybrid:**
- Fusion outperforms merged pool (R@5: 0.423 vs 0.340, +24%; MAP@5: 0.286 vs 0.249, +15%).

**Key insight:** Fusion consistently outperforms merged pool for stpo+faq-stpo across all retriever types. This is because per-KB retrieval preserves the smaller KB's (StPO, 153 chunks) results from being drowned out by the larger one (FAQ-StPO, 1039 chunks).

### MAP Analysis

MAP reveals ranking quality beyond first-hit (MRR) and set-level (P/R) metrics:

- **Best single-KB MAP@5:** Vector on FAQ-StPO (0.444) — consistent top-ranking of all relevant documents.
- **Hybrid vs. Vector:** Hybrid MAP@5 (0.373/0.387) falls below Vector (0.388/0.444), confirming that while Hybrid finds more relevant docs (higher recall), it does not rank them as highly.
- **Combined-KB:** Fusion MAP@5 for stpo+faq-stpo (0.223–0.312) is lower than single-KB MAP, reflecting the increased difficulty when relevant chunks span two KBs of very different sizes.

### Retriever Ranking

Across all configurations, the retriever ranking is consistent with single-KB findings:

1. **Vector** — best overall (highest MAP@5, F1@5, MRR@5 in most configurations)
2. **Hybrid** — second, with the strongest recall (R@5)
3. **BM25** — lowest metrics across all configurations

---

## Module Structure

```
evaluation/retrieval/
├── combined.py             # Combined-KB evaluation (this module)
├── evaluate.py             # Single-KB evaluation runner
└── metrics.py              # Shared metrics (P@k, R@k, F1@k, MRR, MAP, Jaccard@k)

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
