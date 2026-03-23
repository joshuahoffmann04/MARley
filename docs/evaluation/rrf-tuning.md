# RRF k_rrf Parameter Tuning

**Module:** `evaluation/retrieval/rrf_tuning.py`
**Test file:** `evaluation/tests/retrieval/test_rrf_tuning.py`
**Output:** `data/testing/rrf-tuning.json`
**CLI:** `python -m evaluation --rrf-tuning`

This evaluation sweeps the RRF smoothing constant `k_rrf` to determine its impact on retrieval quality. Two sweep types are evaluated: Hybrid (within-KB BM25+Vector fusion) and Fusion (cross-KB BM25 fusion).

---

## Methodology

The sweep evaluates k_rrf ∈ {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100} (11 values). For each value, retrieval is run against all evaluable queries and metrics are computed. The best k_rrf is selected by maximizing Recall@5.

### Hybrid Sweep

For each single KB, a HybridRetriever (BM25 + Vector) is created with the given k_rrf value, indexed on the KB's chunks, and evaluated against its evaluation dataset.

### Fusion Sweep

BM25 retrievers are indexed independently per KB. For each k_rrf value, per-KB results are fused via RRF and evaluated against merged evaluation data.

---

## Results

### Hybrid (BM25 + Vector, within-KB)

| KB | k_rrf | R@5 | MRR@5 | P@5 | Queries |
|---|---|---|---|---|---|
| StPO | all values | 0.547 | 0.432–0.437 | 0.157 | 75 |
| FAQ-StPO | all values | 0.536 | 0.474–0.476 | 0.162 | 74 |
| FAQ-AO | — | 0.000 | 0.000 | 0.000 | 0 |

### Fusion (BM25, cross-KB)

| Combination | k_rrf | R@5 | MRR@5 | P@5 | Queries |
|---|---|---|---|---|---|
| all KBs | all values | 0.294 | 0.374 | 0.163 | 75 |

---

## Analysis

**Recall@5 is completely invariant to k_rrf** across the entire tested range [1, 100] for both Hybrid and Fusion configurations. The same documents appear in the top-5 regardless of the smoothing constant.

**MRR@5 varies by at most ±0.005** (Hybrid/StPO: 0.437 at k_rrf=1 vs. 0.432 at k_rrf≥5). This minimal difference reflects occasional tie-breaking changes at rank 1, with no practical significance.

**Interpretation:** RRF's rank-based fusion is inherently robust to k_rrf because the parameter only controls the relative weight of higher vs. lower ranks. With only two input lists (Hybrid) or a small number of KBs (Fusion), the top-ranked documents from each list dominate regardless of the smoothing curve. The literature default of k_rrf=60 (Cormack et al., 2009) is confirmed as a safe choice without need for per-corpus tuning.

---

## Module Structure

```
evaluation/retrieval/
├── rrf_tuning.py       # Sweep functions: sweep_hybrid_k_rrf(), sweep_fusion_k_rrf()
└── ...

evaluation/tests/retrieval/
└── test_rrf_tuning.py  # 10 unit tests (5 hybrid + 5 fusion)
```
