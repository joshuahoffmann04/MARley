"""Prepare manual evaluation items from E2E pipeline results.

Converts E2EResult instances to EvaluationItem instances for the
Phase 2 manual evaluation UI.
"""

from __future__ import annotations

from evaluation.end_to_end.evaluate import E2EResult
from evaluation.manual.models import EvaluationItem


def prepare_e2e_items(
    results: list[E2EResult],
    config_name: str,
) -> list[EvaluationItem]:
    """Convert E2E pipeline results to manual evaluation items.

    Item IDs follow the format: ``e2e-{config_name}-{question_id}``

    For abstained questions, the generated_answer field contains the
    abstention display text: ``[ABSTENTION Level {n}] {reason}``

    Args:
        results: List of E2EResult from run_e2e_config().
        config_name: Configuration name (used in item ID).

    Returns:
        List of EvaluationItem instances ready for the manual UI.
    """
    items: list[EvaluationItem] = []

    for r in results:
        if r.abstained:
            display_answer = (
                f"[ABSTENTION Level {r.abstention_level}] {r.abstention_reason}"
            )
        else:
            display_answer = r.answer

        item_id = f"e2e-{config_name}-{r.question_id}"

        items.append(EvaluationItem(
            id=item_id,
            question=r.question,
            generated_answer=display_answer,
            reference_answer=r.reference_answer,
            category=r.category,
            expected_abstention=r.expected_abstention,
            metadata={
                "question_id": r.question_id,
                "config_name": config_name,
                "evaluation_type": "end_to_end",
                "abstained": r.abstained,
                "abstention_level": r.abstention_level,
                "confidence": r.confidence,
                "retrieval_chunk_ids": r.retrieval_chunk_ids,
                "generator_model": r.model,
            },
        ))

    return items
