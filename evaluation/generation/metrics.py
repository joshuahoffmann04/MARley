"""Generation evaluation metrics and data classes.

Defines the result types used to track per-question and aggregate
generation evaluation outcomes across distractor levels and quality
dimensions.

Quality metrics per answer:
  ROUGE-1/2/L   -- n-gram overlap with the reference answer (deterministic)
  BERTScore F1  -- semantic similarity via BERT embeddings (model-based)
  faithfulness  -- LLM judge: answer grounded in retrieved context (0-1)
  answer_relevance -- LLM judge: answer addresses the question (0-1)
  correctness   -- LLM judge: answer matches the reference answer (0-1)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationEvalResult:
    """Result of generating an answer for a single question."""

    question_id: str
    num_distractors: int
    generated_answer: str
    reference_answer: str
    context_chunk_ids: list[str] = field(default_factory=list)

    # --- ROUGE (n-gram overlap) ---
    rouge1: float = 0.0
    """ROUGE-1 F1 score (unigram overlap with reference)."""

    rouge2: float = 0.0
    """ROUGE-2 F1 score (bigram overlap with reference)."""

    rougeL: float = 0.0
    """ROUGE-L F1 score (longest common subsequence with reference)."""

    # --- BERTScore (semantic similarity) ---
    bertscore_f1: float = 0.0
    """BERTScore F1 (semantic similarity between generated and reference)."""

    # --- LLM judge scores ---
    faithfulness: float = 0.0
    """LLM judge: answer only uses information from the context (0-1)."""

    answer_relevance: float = 0.0
    """LLM judge: answer addresses the question (0-1)."""

    correctness: float = 0.0
    """LLM judge: answer matches the reference answer (0-1)."""


@dataclass
class GenerationMetrics:
    """Aggregated generation evaluation metrics over a set of results."""

    num_results: int
    results_by_distractors: dict[int, int]
    num_queries: int
    knowledge_base: str
    model: str

    # --- Aggregated ROUGE ---
    avg_rouge1: float = 0.0
    avg_rouge2: float = 0.0
    avg_rougeL: float = 0.0

    # --- Aggregated BERTScore ---
    avg_bertscore_f1: float = 0.0

    # --- Aggregated LLM judge ---
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_correctness: float = 0.0


def compute_generation_metrics(
    results: list[GenerationEvalResult],
    knowledge_base: str,
    model: str,
) -> GenerationMetrics:
    """Compute aggregated metrics from a list of generation results.

    Groups results by distractor count and macro-averages all quality
    scores over all results with non-zero values (abstained answers
    with score 0.0 are included in the average).

    Args:
        results: List of per-question results from run_generation_evaluation.
        knowledge_base: Name of the knowledge base (for metadata).
        model: Generator model identifier (for metadata).

    Returns:
        GenerationMetrics with per-distractor counts and averaged scores.
    """
    if not results:
        return GenerationMetrics(
            num_results=0,
            results_by_distractors={},
            num_queries=0,
            knowledge_base=knowledge_base,
            model=model,
        )

    by_level: dict[int, int] = {}
    for r in results:
        by_level[r.num_distractors] = by_level.get(r.num_distractors, 0) + 1

    n = len(results)
    unique_questions = len({r.question_id for r in results})

    def _avg(attr: str) -> float:
        return sum(getattr(r, attr) for r in results) / n

    return GenerationMetrics(
        num_results=n,
        results_by_distractors=dict(sorted(by_level.items())),
        num_queries=unique_questions,
        knowledge_base=knowledge_base,
        model=model,
        avg_rouge1=_avg("rouge1"),
        avg_rouge2=_avg("rouge2"),
        avg_rougeL=_avg("rougeL"),
        avg_bertscore_f1=_avg("bertscore_f1"),
        avg_faithfulness=_avg("faithfulness"),
        avg_answer_relevance=_avg("answer_relevance"),
        avg_correctness=_avg("correctness"),
    )
