"""Prepare evaluation items from generation result files.

Converts existing generation evaluation result files (which contain
generated answers, reference answers, and metadata for every
(question, distractor_level) pair) into generic EvaluationItem
instances for the manual evaluation UI.

Also loads category and expected_abstention from the original
evaluation dataset files to enrich items with question metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.manual.models import EvaluationItem


def _load_question_metadata(eval_dataset_path: str | Path) -> dict[str, dict]:
    """Load question-level metadata from an evaluation dataset file.

    Returns a dict mapping question_id to {question, category, expected_abstention}.
    """
    path = Path(eval_dataset_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    return {
        q["id"]: {
            "question": q.get("question", ""),
            "category": q.get("category", ""),
            "expected_abstention": q.get("expected_abstention", False),
        }
        for q in data["questions"]
    }


def prepare_generation_items(
    eval_result_path: str | Path,
    knowledge_base: str,
    eval_dataset_path: str | Path | None = None,
) -> list[EvaluationItem]:
    """Convert a generation evaluation result file to evaluation items.

    Reads the existing ``generation-eval-{kb}.json`` file and creates
    one EvaluationItem per (question, distractor_level) pair.

    Item IDs follow the format: ``gen-{kb}-{question_id}-d{num_distractors}``

    Args:
        eval_result_path: Path to the generation result JSON file
            (e.g. ``data/testing/generation-eval-stpo.json``).
        knowledge_base: Knowledge base identifier (e.g. ``"stpo"``).
        eval_dataset_path: Optional path to the original evaluation
            dataset file for enriching items with category and
            expected_abstention. If not provided, defaults to empty
            category and ``False`` for expected_abstention.

    Returns:
        List of EvaluationItem instances.
    """
    path = Path(eval_result_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    # Load question metadata if dataset path provided
    question_meta: dict[str, dict] = {}
    if eval_dataset_path is not None:
        question_meta = _load_question_metadata(eval_dataset_path)

    config = data.get("config", {})
    generator_model = config.get("generator_model", "")

    items: list[EvaluationItem] = []
    for result in data["results"]:
        qid = result["question_id"]
        n_dist = result["num_distractors"]

        meta = question_meta.get(qid, {})
        question_text = meta.get("question", "")
        category = meta.get("category", "")
        expected_abstention = meta.get("expected_abstention", False)

        item_id = f"gen-{knowledge_base}-{qid}-d{n_dist}"

        items.append(EvaluationItem(
            id=item_id,
            question=question_text,
            generated_answer=result["generated_answer"],
            reference_answer=result["reference_answer"],
            category=category,
            expected_abstention=expected_abstention,
            metadata={
                "question_id": qid,
                "knowledge_base": knowledge_base,
                "num_distractors": n_dist,
                "context_chunk_ids": result.get("context_chunk_ids", []),
                "evaluation_type": "generation",
                "generator_model": generator_model,
            },
        ))

    return items


def prepare_items_from_results(
    results: list[dict],
    knowledge_base: str,
    generator_model: str,
    question_metadata: dict[str, dict] | None = None,
) -> list[EvaluationItem]:
    """Convert raw generation result dicts to evaluation items.

    Used for future generation runs (Phase 3, Phase 5) where results
    are produced programmatically rather than loaded from a file.

    Args:
        results: List of result dicts with at minimum ``question_id``,
            ``num_distractors``, ``generated_answer``, ``reference_answer``.
        knowledge_base: Knowledge base identifier.
        generator_model: Model name used for generation.
        question_metadata: Optional mapping of question_id to
            ``{category, expected_abstention}``.

    Returns:
        List of EvaluationItem instances.
    """
    question_metadata = question_metadata or {}

    items: list[EvaluationItem] = []
    for result in results:
        qid = result["question_id"]
        n_dist = result["num_distractors"]

        meta = question_metadata.get(qid, {})
        question_text = meta.get("question", "")
        category = meta.get("category", "")
        expected_abstention = meta.get("expected_abstention", False)

        item_id = f"gen-{knowledge_base}-{qid}-d{n_dist}"

        items.append(EvaluationItem(
            id=item_id,
            question=question_text,
            generated_answer=result["generated_answer"],
            reference_answer=result["reference_answer"],
            category=category,
            expected_abstention=expected_abstention,
            metadata={
                "question_id": qid,
                "knowledge_base": knowledge_base,
                "num_distractors": n_dist,
                "context_chunk_ids": result.get("context_chunk_ids", []),
                "evaluation_type": "generation",
                "generator_model": generator_model,
            },
        ))

    return items
