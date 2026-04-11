# Generation Evaluation Results

> Generated: 2026-04-11 | Model: llama3.1:latest (8B, Q4_K_M) | RAGAS 0.4.3
> 75 answerable questions per KB x 11 distractor levels (0-10) = 825 samples per KB

## Overview

Generation quality was evaluated using three RAGAS metrics scored by the same
local Ollama model (llama3.1:latest). Each answerable question was tested with
0 to 10 BM25-ranked distractors mixed into the context, measuring robustness
against retrieval noise.

**Failed RAGAS samples** (LLM producing invalid structured output) were excluded
from averages. Failure rates were low (1-4% per metric).

## Results by Knowledge Base

### stpo (825 samples, 21/32/8 NaN in faith/relev/correct)

| Distractors | Faithfulness | Answer Relevancy | Factual Correctness |
| ----------- | ------------ | ---------------- | ------------------- |
| 0 (ideal)   | 0.432        | 0.296            | 0.406               |
| 5           | 0.383        | 0.315            | 0.321               |
| 10 (noisy)  | 0.423        | 0.413            | 0.329               |
| **Overall** | **0.396**    | **0.318**        | **0.351**           |

### faq-stpo (825 samples, 4/14/1 NaN in faith/relev/correct)

| Distractors | Faithfulness | Answer Relevancy | Factual Correctness |
| ----------- | ------------ | ---------------- | ------------------- |
| 0 (ideal)   | 0.517        | 0.272            | 0.364               |
| 5           | 0.446        | 0.317            | 0.354               |
| 10 (noisy)  | 0.414        | 0.277            | 0.378               |
| **Overall** | **0.434**    | **0.292**        | **0.360**           |

## Key Findings

1. **Overall quality is modest** (0.29-0.43 across all metrics), reflecting the
   limitations of a local 8B quantized model for study advising in German. This
   is a baseline — larger models or fine-tuning would likely improve scores.

2. **Faithfulness decreases with more distractors** on faq-stpo (0.517 -> 0.414),
   confirming that retrieval noise causes the LLM to incorporate irrelevant context.
   The effect is less pronounced on stpo.

3. **Factual correctness also degrades** with distractors on stpo (0.406 -> 0.329),
   showing that noisy context leads to less accurate answers.

4. **faq-stpo outperforms stpo on faithfulness** (0.434 vs 0.396), consistent with
   retrieval results — FAQ-formatted chunks provide clearer, more self-contained
   context for the generator.

5. **Answer relevancy shows unexpected patterns** — scores sometimes increase with
   more distractors. This may reflect RAGAS scoring artifacts with the local model
   (relevancy scoring uses embeddings + LLM, making it sensitive to response length).

6. **NaN rates are low** (<4%), validating the chunked batch scoring approach with
   per-sample retry fallback.

## Methodology Notes

- **RAGAS version**: 0.4.3 with InstructorLLM via Ollama OpenAI-compatible API
- **Scoring**: Chunked batch scoring (batch_size=10) with up to 3 retries per failed sample
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (for Answer Relevancy)
- **Distractor selection**: BM25-ranked non-relevant chunks (hardest distractors first)
- **Context assembly**: Deterministic shuffle with fixed seed per question
