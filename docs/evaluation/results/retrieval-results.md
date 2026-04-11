# Retrieval Evaluation Results

> Generated: 2026-04-10 | Dataset: 75 answerable queries per KB | k=5

## Overview

The retrieval evaluation measures five metrics across three retriever types (BM25, Vector, Hybrid),
two knowledge bases (stpo, faq-stpo), and two multi-KB strategies (merged pool, fusion).

**Metrics:**
- **P@5** (Precision@5): Fraction of retrieved chunks that are relevant
- **R@5** (Recall@5): Fraction of relevant chunks that are retrieved
- **MRR** (Mean Reciprocal Rank): Average reciprocal rank of the first relevant chunk
- **F1@5**: Harmonic mean of P@5 and R@5
- **J@5** (Jaccard@5): Intersection-over-union of retrieved and relevant sets

## Single-KB Results

### stpo (153 chunks, StPO-extracted)

| Retriever | P@5 | R@5 | MRR | F1@5 | J@5 |
|-----------|-------|-------|-------|-------|-------|
| BM25 | 0.115 | 0.413 | 0.360 | 0.175 | 0.109 |
| Vector | 0.152 | 0.494 | 0.474 | 0.225 | 0.144 |
| Hybrid | 0.160 | 0.540 | 0.446 | 0.239 | 0.151 |

### faq-stpo (1039 chunks, FAQ dataset)

| Retriever | P@5 | R@5 | MRR | F1@5 | J@5 |
|-----------|-------|-------|-------|-------|-------|
| BM25 | 0.131 | 0.402 | 0.383 | 0.190 | 0.116 |
| Vector | 0.184 | 0.557 | 0.557 | 0.267 | 0.170 |
| Hybrid | 0.168 | 0.533 | 0.497 | 0.248 | 0.156 |

## Multi-KB Results (all KBs combined, 1192 chunks)

### Merged Pool Strategy

Merges all chunks into a single corpus before retrieval.

| Retriever | P@5 | R@5 | MRR | F1@5 | J@5 |
|-----------|-------|-------|-------|-------|-------|
| BM25 | 0.139 | 0.237 | 0.392 | 0.168 | 0.103 |
| Vector | 0.243 | 0.383 | 0.560 | 0.286 | 0.188 |
| Hybrid | 0.200 | 0.340 | 0.487 | 0.244 | 0.155 |

### Fusion Strategy (RRF, k_rrf=60)

Retrieves from each KB independently and fuses results via Reciprocal Rank Fusion.

| Retriever | P@5 | R@5 | MRR | F1@5 | J@5 |
|-----------|-------|-------|-------|-------|-------|
| BM25 | 0.171 | 0.307 | 0.394 | 0.213 | 0.140 |
| Vector | 0.264 | 0.422 | 0.549 | 0.313 | 0.211 |
| Hybrid | 0.253 | 0.423 | 0.478 | 0.306 | 0.201 |

## RRF k-Parameter Tuning

A sweep over k_rrf in {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100} shows
minimal sensitivity to this parameter:

| Sweep | KB | Best k_rrf | MRR | R@5 | F1@5 |
|-------|---------|------------|-------|-------|-------|
| Hybrid | stpo | 1 | 0.450 | 0.540 | 0.239 |
| Hybrid | faq-stpo | 1 | 0.494 | 0.533 | 0.248 |
| Fusion | all | 1 | 0.394 | 0.307 | 0.213 |

**Finding:** k_rrf has negligible impact on retrieval quality. Across all configurations,
P@5, R@5, and F1@5 remain constant regardless of k_rrf. Only MRR shows a marginal
difference at k_rrf=1 vs. higher values (delta < 0.005). This is consistent with
RRF theory: k_rrf primarily controls the smoothness of rank-score mapping and has
less effect when the number of retrieved results is small (k=5).

## Key Findings

1. **Vector retrieval consistently outperforms BM25** across all KBs and strategies,
   demonstrating that semantic similarity (all-mpnet-base-v2 embeddings) captures
   study advising query intent better than keyword matching.

2. **Hybrid retrieval improves recall** over pure Vector on stpo (0.540 vs. 0.494)
   but not on faq-stpo, suggesting that BM25 complements embeddings primarily for
   shorter, more structured documents.

3. **Fusion strategy yields the highest precision and F1** for multi-KB retrieval
   (Vector Fusion: P@5=0.264, F1@5=0.313), outperforming merged pool. Retrieving
   per-KB and fusing preserves KB-specific ranking quality.

4. **Merged pool achieves highest MRR** (Vector: 0.560), indicating that merging
   corpora helps surface the single most relevant chunk, even if overall set quality
   (F1, Jaccard) is lower than fusion.

5. **faq-stpo outperforms stpo** on single-KB metrics despite having 7x more chunks,
   confirming that the FAQ format (explicit Q&A pairs) aligns better with evaluation
   queries than raw StPO text chunks.

6. **Precision is inherently low** (0.11-0.26) because k=5 retrieves 5 chunks while
   most queries have 1-3 relevant chunks, making a theoretical maximum P@5 of 0.2-0.6.

7. **RRF k-parameter is not a significant tuning lever** for this dataset.
   The default k_rrf=60 performs equivalently to k_rrf=1.
