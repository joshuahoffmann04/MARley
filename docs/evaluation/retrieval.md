# Retrieval Evaluation

> Evaluates chunk retrieval quality across single-KB, merged pool, and fusion strategies.

## Metrics

Five hand-coded metrics evaluate retrieval quality at cutoff `k` (default: 5).

### Precision@k



Measures the fraction of retrieved chunks that are actually relevant. A Precision@5 of 0.8 means 4 out of 5 retrieved chunks are relevant.

### Recall@k



Measures the fraction of all relevant chunks that were successfully retrieved. With 3 relevant chunks total, retrieving 2 yields Recall@5 = 0.67.

### MRR (Mean Reciprocal Rank)



Measures how quickly the first relevant chunk appears. If the first relevant chunk is at position 1, MRR = 1.0. At position 3, MRR = 0.33. Averaged over all queries.

### F1@k



Harmonic mean of Precision@k and Recall@k. Balances the trade-off between retrieving many relevant chunks (recall) and avoiding irrelevant ones (precision).

### Jaccard@k



Set similarity between relevant and retrieved chunks. Unlike F1@k, Jaccard penalizes both missed relevant chunks and retrieved irrelevant ones equally.

### Aggregation

All metrics are computed per-query and then **macro-averaged** over all answerable queries. Questions marked as `expected_abstention=True` are skipped (they have no relevant chunks by definition).

Implementation: `evaluation/retrieval/metrics.py`



## Evaluation Strategies

### Single-KB Evaluation

Evaluates each retriever type against each knowledge base independently.

**Matrix**: 3 retriever types × 3 KBs = 9 configurations

| Retriever | Knowledge Bases |
|---|---|
| BM25Retriever | stpo, faq-stpo, faq-ao |
| VectorRetriever | stpo, faq-stpo, faq-ao |
| HybridRetriever | stpo, faq-stpo, faq-ao |

**Process**:
1. Load chunks from the KB's chunk file
2. Index the retriever on those chunks
3. Run all evaluation questions against the retriever
4. Compute metrics at k=5

Implementation: `evaluation/retrieval/evaluate.py` (`run_evaluation()`, `run_and_report()`)

### Merged Pool Evaluation

Combines chunks from multiple KBs into a single corpus and indexes one retriever.

**Process**:
1. Load and merge chunks from all KBs (`merge_chunks()` — validates no duplicate chunk IDs)
2. Index a single retriever on the merged corpus
3. Merge evaluation data (`merge_evaluation_data()` — union of `relevant_chunks` per question)
4. Evaluate with standard metrics

**Combination evaluated**: stpo + faq-stpo + faq-ao (all 3 KBs)

Implementation: `evaluation/retrieval/combined.py` (`run_merged_pool_evaluation()`)

### Fusion Evaluation

Each KB gets its own retriever instance; per-query results are fused via Reciprocal Rank Fusion (RRF).

**Process**:
1. Create one retriever per KB using a `retriever_factory` callable
2. Index each retriever on its own KB's chunks
3. Merge evaluation data (same as merged pool)
4. For each query: retrieve from all KB retrievers, fuse via `rrf_fuse()`
5. Evaluate fused results with standard metrics

**Parameters**:
- `k_rrf`: RRF smoothing constant (default: 60)
- `k`: Number of top results to consider (default: 5)

Implementation: `evaluation/retrieval/combined.py` (`run_fusion_evaluation()`)

### Strategy Comparison

| Aspect | Single-KB | Merged Pool | Fusion |
|---|---|---|---|
| Indexing | One retriever per KB | One retriever, merged corpus | One retriever per KB |
| Query handling | Per-KB queries | Single query on merged index | Per-KB queries + RRF fusion |
| Cross-KB coverage | None | Implicit (merged index) | Explicit (RRF across KBs) |
| Use case | Baseline per KB | Simple multi-KB | Controlled multi-KB fusion |

## RRF Tuning

The RRF smoothing constant `k_rrf` controls how much rank position influences the fused score. Tuning sweeps across a range of values to find the optimal `k_rrf`.

### Sweep Values

Default sweep range: `[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]`

### Hybrid Sweep

Tunes `k_rrf` for `HybridRetriever` (BM25 + Vector within a single KB).

**Process**:
1. For each `k_rrf` value, create a fresh `HybridRetriever(BM25, Vector, k_rrf=k_rrf)`
2. Index on the KB's chunks
3. Evaluate retrieval quality
4. Select the `k_rrf` that maximizes Recall@k

Implementation: `evaluation/retrieval/rrf_tuning.py` (`sweep_hybrid_k_rrf()`)

### Fusion Sweep

Tunes `k_rrf` for cross-KB fusion (FusionRetriever-style evaluation).

**Process**:
1. Build and index one retriever per KB (shared across all sweep values)
2. For each `k_rrf` value, run per-query fusion with `rrf_fuse(result_lists, k_rrf=k_rrf)`
3. Evaluate fused results
4. Select the `k_rrf` that maximizes Recall@k

Implementation: `evaluation/retrieval/rrf_tuning.py` (`sweep_fusion_k_rrf()`)

### Output Format



## CLI Usage



## Output Files

| File | Content |
|---|---|
| `retrieval-evaluation.json` | All single-KB, merged pool, and fusion results |
| `rrf-tuning.json` | Hybrid and fusion sweep results with best `k_rrf` |
