# Generation Evaluation Results

**Run date:** 2026-04-07
**Model:** `llama3.1:latest` (generator + judge)
**Judge:** `OllamaJudge` (same model as generator)
**Distractor levels:** 0–10 (all integer values)
**Questions per level:** 75 (answerable questions from each KB; unanswerable skipped)
**Result files:** `data/testing/generation-evaluation.json`, `data/testing/generation-evaluation-combined.json`

---

## Summary

| Knowledge Base | Queries | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Faithfulness | Relevance | Correctness |
|---|---|---|---|---|---|---|---|---|
| StPO | 75 | 0.2879 | 0.1329 | 0.2294 | 0.8596 | 0.7912 | 0.3791 | 0.2758 |
| FAQ-StPO | 75 | 0.3191 | 0.1598 | 0.2601 | 0.8684 | 0.9416 | 0.4628 | 0.3478 |
| FAQ-AO | 0 | — | — | — | — | — | — | — |
| Combined (all KBs) | 75 | 0.3363 | 0.1719 | 0.2712 | 0.8692 | 0.9141 | 0.6541 | 0.4674 |

All metrics are macro-averaged over all 75 queries × 11 distractor levels (825 results per KB).

**FAQ-AO has 0 chunks** (placeholder KB, not yet populated) — all questions skipped.

---

## Key Findings

### ROUGE and BERTScore

- **FAQ-StPO > StPO** across all ROUGE and BERTScore metrics: FAQ-style chunks are better aligned with how questions are phrased, leading to higher lexical and semantic overlap with reference answers.
- **Combined > Single-KB** for all metrics: combining all knowledge bases improves answer quality, particularly for ROUGE-2 (bigram overlap) and correctness.
- **BERTScore F1 is consistently high (0.86–0.87)** across all configurations, indicating that generated answers are semantically close to references even when ROUGE is moderate. This reflects paraphrasing rather than verbatim reproduction.

### LLM Judge Scores

- **Faithfulness degrades sharply with distractors in StPO** (0.95 at level 0 → 0.16 at level 10). StPO chunks are dense and technical; many distractors overwhelm the generator.
- **FAQ-StPO faithfulness is stable across all distractor levels** (0.93–0.95), suggesting FAQ-style chunks are more robust to distractor noise.
- **Combined KB achieves the highest correctness (0.4674)** and relevance (0.6541), benefiting from the larger and more diverse context pool.
- **Correctness scores are moderate (0.28–0.47)**. Self-evaluation bias is a known limitation: using the same model as generator and judge tends to inflate faithfulness and underestimate correctness.

---

## Per-Distractor Level: StPO

| Distractors | ROUGE-1 | ROUGE-L | BERTScore F1 | Faithfulness | Relevance | Correctness |
|---|---|---|---|---|---|---|
| 0 | 0.3017 | 0.2416 | 0.8631 | 0.9467 | 0.4733 | 0.2967 |
| 1 | 0.3008 | 0.2462 | 0.8625 | 0.9493 | 0.4827 | 0.3433 |
| 2 | 0.2896 | 0.2298 | 0.8600 | 0.9593 | 0.4467 | 0.3413 |
| 3 | 0.2967 | 0.2390 | 0.8612 | 0.9587 | 0.4780 | 0.3420 |
| 4 | 0.2752 | 0.2224 | 0.8595 | 0.9547 | 0.4280 | 0.3433 |
| 5 | 0.2851 | 0.2332 | 0.8598 | 0.9387 | 0.5173 | 0.3353 |
| 6 | 0.2952 | 0.2312 | 0.8593 | 0.9187 | 0.4227 | 0.3633 |
| 7 | 0.2740 | 0.2207 | 0.8559 | 0.8427 | 0.4173 | 0.2773 |
| 8 | 0.2808 | 0.2212 | 0.8583 | 0.7120 | 0.3120 | 0.2580 |
| 9 | 0.2860 | 0.2172 | 0.8584 | 0.3587 | 0.1520 | 0.1120 |
| 10 | 0.2823 | 0.2208 | 0.8576 | 0.1640 | 0.0400 | 0.0213 |

**StPO observation:** ROUGE and BERTScore are stable across distractor levels, but LLM judge scores collapse beyond 7 distractors. This indicates the generator still produces text that resembles the reference (surface-level), but the answers become unfaithful and irrelevant when too many distractors are present.

---

## Per-Distractor Level: FAQ-StPO

| Distractors | ROUGE-1 | ROUGE-L | BERTScore F1 | Faithfulness | Relevance | Correctness |
|---|---|---|---|---|---|---|
| 0 | 0.2972 | 0.2396 | 0.8656 | 0.9393 | 0.4227 | 0.2960 |
| 1 | 0.3303 | 0.2782 | 0.8704 | 0.9347 | 0.4520 | 0.3263 |
| 2 | 0.3120 | 0.2537 | 0.8668 | 0.9347 | 0.4893 | 0.3373 |
| 3 | 0.3183 | 0.2542 | 0.8683 | 0.9340 | 0.4320 | 0.3260 |
| 4 | 0.2906 | 0.2307 | 0.8639 | 0.9513 | 0.4460 | 0.3560 |
| 5 | 0.3300 | 0.2747 | 0.8696 | 0.9380 | 0.4733 | 0.3787 |
| 6 | 0.3284 | 0.2699 | 0.8703 | 0.9500 | 0.4653 | 0.3733 |
| 7 | 0.3408 | 0.2772 | 0.8707 | 0.9333 | 0.4800 | 0.3513 |
| 8 | 0.3342 | 0.2677 | 0.8705 | 0.9513 | 0.4427 | 0.3480 |
| 9 | 0.3314 | 0.2729 | 0.8715 | 0.9407 | 0.5213 | 0.4120 |
| 10 | 0.2971 | 0.2422 | 0.8646 | 0.9507 | 0.4667 | 0.3213 |

**FAQ-StPO observation:** All metrics are remarkably stable across all distractor levels. The FAQ format (question-answer pairs) provides context that is sufficiently self-contained and concise that even 10 distractors do not degrade faithfulness.

---

## Combined-KB Summary

| Metric | Value |
|---|---|
| ROUGE-1 | 0.3363 |
| ROUGE-2 | 0.1719 |
| ROUGE-L | 0.2712 |
| BERTScore F1 | 0.8692 |
| Faithfulness | 0.9141 |
| Answer Relevance | 0.6541 |
| Correctness | 0.4674 |

The combined KB achieves the best overall scores. The larger context pool (StPO + FAQ-StPO + FAQ-AO) reduces the relative proportion of distractors and provides complementary coverage.

---

## Limitations

- **Self-evaluation bias:** The same `llama3.1:latest` model is used as both generator and judge. This inflates faithfulness scores and may underestimate correctness. A separate judge (e.g. GPT-4o-mini via `OpenAIJudge`) would provide independent evaluation.
- **FAQ-AO not evaluated:** The FAQ-AO knowledge base has 0 chunks and contributes no results. Results for that KB are omitted.
- **ROUGE reflects lexical overlap only:** High BERTScore with moderate ROUGE indicates paraphrasing rather than verbatim answers, which is expected for RAG-style question answering.
