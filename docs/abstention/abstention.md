# Abstention Pipeline

`src/marley/abstention/` — Two-level abstention mechanism for the MARley RAG pipeline.

The abstention module prevents the system from returning unreliable or
fabricated answers. It operates at two levels: retrieval confidence
filtering (Level 1) and LLM output detection (Level 2). Together, these
levels ensure that the pipeline either returns a well-grounded answer or
explicitly abstains with a reason.

---

## Theoretical Background

Abstention (also called *selective prediction* or *refusal*) is the ability of a system to decline answering when it cannot produce a reliable response. In the context of question answering, abstention addresses the fundamental problem of **hallucination** — the tendency of language models to generate plausible but factually incorrect answers when the available evidence is insufficient (Ji et al., 2023).

Two general approaches to abstention exist in RAG systems:

1. **Retrieval-side abstention** evaluates whether the retrieved context is likely to contain the answer. If retrieval confidence is below a threshold, the system abstains before invoking the generator, saving computation and avoiding unnecessary hallucination risk.
2. **Generation-side abstention** relies on the language model itself to recognize when the provided context is insufficient. This is typically implemented through prompt engineering — instructing the model to produce a structured abstention signal rather than guessing.

MARley implements both levels as complementary safeguards. Level 1 (retrieval confidence) uses score normalization to map retriever-specific scores to a common [0, 1] range, enabling a single threshold to work across BM25, Vector, and RRF retrieval strategies. Level 2 (LLM detection) uses a structured `ABSTENTION: <reason>` prefix that enables deterministic detection via string matching, avoiding the fragility of heuristic-based approaches.

This two-level design follows the defense-in-depth principle: Level 1 catches cases where retrieval clearly fails, while Level 2 catches subtler cases where passages pass the threshold but do not actually address the question.

---

## Overview

A Retrieval-Augmented Generation pipeline can fail silently when the
retrieved context is irrelevant or insufficient. Without an abstention
mechanism, the language model may hallucinate plausible-sounding answers
that are not grounded in the knowledge base.

MARley addresses this with a two-level abstention strategy:

| Level | Name | Mechanism | Trigger |
|-------|------|-----------|---------|
| 1 | Retrieval Confidence | Score normalization + threshold filtering | No passage meets the confidence threshold |
| 2 | LLM Detection | Structured output parsing | Model responds with `ABSTENTION: <reason>` |

Level 1 acts as a pre-generation gate: if no retrieved passage is
sufficiently relevant, the pipeline abstains without invoking the LLM.
Level 2 catches cases where the context passes the threshold but the
model itself determines that the information is insufficient to answer
the question.

---

## System Prompt

The generator uses an abstention-aware system prompt that instructs the
LLM to produce a structured abstention signal when it cannot answer from
the provided context.

```
You are a study advisor for the M.Sc. Computer Science program
at Philipps-Universität Marburg.

Answer the student's question using ONLY the numbered context
passages below. Follow these rules:

1. Base your answer exclusively on information from the provided
   context passages.
2. Be concise, precise, and factually accurate.
3. If the provided context does not contain sufficient information
   to fully answer the question, respond with exactly:
   ABSTENTION: <reason>
4. Never guess, speculate, or supplement with knowledge not present
   in the context.
```

### Design Rationale

The new prompt resolves several weaknesses of the original prompt design:

| Aspect | Old prompt | New prompt |
|---|---|---|
| Abstention signal | Vague ("state that clearly") | Structured (`ABSTENTION: <reason>`) |
| Detection | Impossible (heuristic) | Deterministic (prefix match) |
| Scope restriction | "based ONLY" | "exclusively on information from the provided context passages" |
| Guessing | Not addressed | Explicitly forbidden (rule 4) |
| Context reference | Generic | "numbered context passages" (connects to `[1]`, `[2]`, ... format) |

The structured `ABSTENTION: <reason>` format enables deterministic
detection via a simple prefix match, eliminating the need for fragile
heuristics or secondary LLM calls.

---

## Level 1: Retrieval Confidence

Level 1 operates before generation. It normalizes retrieval scores to a
common `[0, 1]` scale and filters passages that fall below a confidence
threshold.

### Score Normalization

Each retriever produces scores on a different scale. Normalization maps
them into a uniform range so that a single threshold applies across all
retrieval strategies.

| Strategy | Formula | Rationale |
|----------|---------|-----------|
| BM25 | `score / (score + k)` | Saturation function; unbounded BM25 scores are compressed into `[0, 1)`. Parameter `k` controls the midpoint (default `k=1.0`). |
| Vector | identity | Cosine similarity already lies in `[0, 1]`; no transformation needed. |
| RRF | `score * (k_rrf + 1) / n_retrievers` | Normalizes by the theoretical maximum RRF score. `k_rrf` is the RRF smoothing constant, `n_retrievers` is the number of fused retrievers. |

### BM25 Normalization Examples (k=1.0)

The saturation function `f(s) = s / (s + k)` maps raw BM25 scores to
normalized confidence values:

| Raw BM25 score | Normalized score |
|----------------|------------------|
| 0.5 | 0.33 |
| 1.0 | 0.50 |
| 5.0 | 0.83 |
| 10.0 | 0.91 |
| 20.0 | 0.95 |

Higher raw scores converge toward 1.0 but never reach it, which
prevents any single high-scoring passage from dominating the confidence
signal.

### Threshold Filtering

After normalization, passages with a normalized score below the
configured threshold are discarded. The retrieval confidence is defined
as the maximum normalized score among the remaining passages:

```
confidence = max(normalized_scores)  # after filtering
```

If no passages survive filtering, the pipeline abstains at Level 1
without invoking the generator.

---

## Level 2: LLM Detection

Level 2 operates after generation. It inspects the LLM output for the
structured abstention signal.

### Detection Logic

The `detect_abstention()` function performs a case-insensitive prefix
match on the generated answer:

```python
def detect_abstention(answer: str) -> bool:
    """Detect whether the LLM output is an abstention.

    Returns True if the answer starts with 'ABSTENTION:'
    (case-insensitive, ignoring leading whitespace).
    """
    return answer.strip().upper().startswith("ABSTENTION:")


def extract_abstention_reason(answer: str) -> str:
    """Extract the reason from an abstention response.

    Returns the text after 'ABSTENTION:', stripped of whitespace.
    Returns an empty string if the answer is not an abstention.
    """
    stripped = answer.strip()
    if not stripped.upper().startswith("ABSTENTION:"):
        return ""
    return stripped[len("ABSTENTION:"):].strip()
```

The two-function design separates detection from extraction: callers
that only need to know *whether* the model abstained use
`detect_abstention()`, while callers that also need the *reason* call
`extract_abstention_reason()`. No fuzzy matching or secondary
classification is required.

---

## Pipeline

The `run_with_abstention()` function orchestrates both levels into a
single pipeline call.

### Data Flow

```
Query
  │
  ├── Retriever.retrieve(query, k) ──> RetrievalResults
  ├── normalize_scores(results) ──> NormalizedResults
  ├── filter_by_threshold(results, threshold) ──> FilteredResults
  ├── Level 1: len(FilteredResults) == 0?
  │   ├── YES ──> ABSTAIN (Level 1)
  │   └── NO ──> continue
  ├── Generator.generate(query, FilteredResults) ──> Answer
  ├── Level 2: detect_abstention(answer)?
  │   ├── YES ──> ABSTAIN (Level 2)
  │   └── NO ──> ANSWER
  └── AbstentionResult
```

### Behavior Summary

| Scenario | Level 1 | Level 2 | Result |
|----------|---------|---------|--------|
| No relevant passages | Triggers | Skipped | Abstention (retrieval) |
| Relevant passages, sufficient context | Passes | Passes | Answer returned |
| Relevant passages, insufficient context | Passes | Triggers | Abstention (generation) |

---

## Configuration

The abstention module is configured through the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.3` | Minimum normalized score for a passage to be considered relevant. |
| `normalization_strategy` | `str` | `"vector"` | Score normalization strategy: `"bm25"`, `"vector"`, or `"rrf"`. Select based on the retriever type (see [Retrieval Overview](../retrieval/overview.md)). |
| `bm25_k` | `float` | `1.0` | Saturation parameter for BM25 score normalization. |
| `rrf_k` | `int` | `60` | RRF smoothing constant, must match the value used in the RRF retriever. |
| `n_retrievers` | `int` | `2` | Number of retrievers fused in RRF, used for RRF score normalization. |

---

## Data Model

The pipeline returns an `AbstentionResult` dataclass that encapsulates
the full outcome of a query, including whether abstention occurred and
at which level.

```python
@dataclass
class AbstentionResult:
    """Result of the abstention-aware pipeline."""

    abstained: bool
    """Whether the pipeline abstained from answering."""

    level: int | None
    """Abstention level: 1 = retrieval confidence, 2 = LLM detection,
    None = answered."""

    reason: str
    """Human-readable abstention reason (empty string if answered)."""

    answer: str
    """Generated answer text (empty string if abstained)."""

    confidence: float
    """Top-1 normalized retrieval score in [0, 1]."""

    retrieval_results: list[dict]
    """Retrieval results passed to (or filtered before) generation.
    Each dict has 'chunk_id', 'text', and 'score'."""

    model: str
    """Model identifier of the generator used (empty if abstained at Level 1)."""
```

---

## Module Structure

```
src/marley/
├── models/
│   └── abstention.py              AbstentionResult dataclass
├── generator/
│   └── prompt.py                  Abstention-aware system prompt
└── abstention/
    ├── __init__.py                Public API exports
    ├── confidence.py              Score normalization + threshold filtering
    ├── detection.py               LLM output abstention detection
    └── pipeline.py                run_with_abstention() orchestrator
```

---

## See Also

- [Abstention Evaluation](../evaluation/abstention.md) — Metrics and
  evaluation methodology for abstention quality.
- [Pipeline Overview](../evaluation/overview.md) — End-to-end pipeline
  architecture and module relationships.
