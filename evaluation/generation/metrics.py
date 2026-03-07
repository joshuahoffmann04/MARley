"""Generation evaluation metrics and data classes.

Defines the result types used to track per-question and aggregate
generation evaluation outcomes across different distractor levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationEvalResult:
    """Result of evaluating a single question at a specific distractor level."""

    question_id: str
    num_distractors: int
    correct: bool
    confidence: float
    generated_answer: str
    reference_answer: str
    judge_reasoning: str
    context_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GenerationMetrics:
    """Aggregated generation evaluation metrics."""

    accuracy: float
    accuracy_by_distractors: dict[int, float]
    num_queries: int
    knowledge_base: str
    model: str
    judge_model: str


def compute_generation_metrics(
    results: list[GenerationEvalResult],
    knowledge_base: str,
    model: str,
    judge_model: str,
) -> GenerationMetrics:
    """Compute aggregated metrics from a list of evaluation results.

    Groups results by distractor count and computes accuracy for each
    level as well as overall accuracy.
    """
    if not results:
        return GenerationMetrics(
            accuracy=0.0,
            accuracy_by_distractors={},
            num_queries=0,
            knowledge_base=knowledge_base,
            model=model,
            judge_model=judge_model,
        )

    # Group by distractor count
    by_level: dict[int, list[bool]] = {}
    for r in results:
        by_level.setdefault(r.num_distractors, []).append(r.correct)

    accuracy_by_distractors = {
        level: sum(outcomes) / len(outcomes)
        for level, outcomes in sorted(by_level.items())
    }

    all_correct = [r.correct for r in results]
    # Count unique question IDs
    unique_questions = len({r.question_id for r in results})

    return GenerationMetrics(
        accuracy=sum(all_correct) / len(all_correct),
        accuracy_by_distractors=accuracy_by_distractors,
        num_queries=unique_questions,
        knowledge_base=knowledge_base,
        model=model,
        judge_model=judge_model,
    )
