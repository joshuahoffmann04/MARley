"""Generation evaluation metrics and data classes.

Defines the result types used to track per-question and aggregate
generation evaluation outcomes across different distractor levels.

Correctness assessment is handled separately by the manual evaluation
framework (``evaluation.manual``). The metrics here track generation
outputs without judging correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationEvalResult:
    """Result of generating an answer for a single question at a specific distractor level."""

    question_id: str
    num_distractors: int
    generated_answer: str
    reference_answer: str
    context_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GenerationMetrics:
    """Aggregated generation evaluation metrics."""

    num_results: int
    results_by_distractors: dict[int, int]
    num_queries: int
    knowledge_base: str
    model: str


def compute_generation_metrics(
    results: list[GenerationEvalResult],
    knowledge_base: str,
    model: str,
) -> GenerationMetrics:
    """Compute aggregated metrics from a list of generation results.

    Groups results by distractor count and reports counts per level
    as well as the total number of unique questions evaluated.
    """
    if not results:
        return GenerationMetrics(
            num_results=0,
            results_by_distractors={},
            num_queries=0,
            knowledge_base=knowledge_base,
            model=model,
        )

    # Group by distractor count
    by_level: dict[int, int] = {}
    for r in results:
        by_level[r.num_distractors] = by_level.get(r.num_distractors, 0) + 1

    results_by_distractors = dict(sorted(by_level.items()))

    # Count unique question IDs
    unique_questions = len({r.question_id for r in results})

    return GenerationMetrics(
        num_results=len(results),
        results_by_distractors=results_by_distractors,
        num_queries=unique_questions,
        knowledge_base=knowledge_base,
        model=model,
    )
