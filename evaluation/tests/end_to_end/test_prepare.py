"""Tests for end-to-end evaluation item preparation."""

from __future__ import annotations

from evaluation.end_to_end.evaluate import E2EResult
from evaluation.end_to_end.prepare import prepare_e2e_items


def _result(
    question_id: str = "q1",
    question: str = "What is X?",
    reference_answer: str = "X is Y.",
    category: str = "direct",
    expected_abstention: bool = False,
    answer: str = "The answer.",
    abstained: bool = False,
    abstention_level: int | None = None,
    abstention_reason: str = "",
    confidence: float = 0.8,
    retrieval_chunk_ids: list[str] | None = None,
    model: str = "stub-model",
) -> E2EResult:
    return E2EResult(
        question_id=question_id,
        question=question,
        reference_answer=reference_answer,
        category=category,
        expected_abstention=expected_abstention,
        answer=answer,
        abstained=abstained,
        abstention_level=abstention_level,
        abstention_reason=abstention_reason,
        confidence=confidence,
        retrieval_chunk_ids=retrieval_chunk_ids or ["c1"],
        model=model,
    )


class TestPrepareE2EItems:
    """Tests for prepare_e2e_items()."""

    def test_answered_question_maps_correctly(self):
        items = prepare_e2e_items([_result()], "test-cfg")
        assert len(items) == 1
        assert items[0].generated_answer == "The answer."

    def test_level1_abstention_display_text(self):
        r = _result(
            abstained=True,
            abstention_level=1,
            abstention_reason="retrieval confidence below threshold",
            answer="",
        )
        items = prepare_e2e_items([r], "cfg")
        assert items[0].generated_answer == (
            "[ABSTENTION Level 1] retrieval confidence below threshold"
        )

    def test_level2_abstention_display_text(self):
        r = _result(
            abstained=True,
            abstention_level=2,
            abstention_reason="insufficient context",
            answer="",
        )
        items = prepare_e2e_items([r], "cfg")
        assert items[0].generated_answer == (
            "[ABSTENTION Level 2] insufficient context"
        )

    def test_item_id_format(self):
        items = prepare_e2e_items([_result(question_id="q42")], "my-config")
        assert items[0].id == "e2e-my-config-q42"

    def test_metadata_fields_complete(self):
        items = prepare_e2e_items([_result()], "cfg")
        meta = items[0].metadata
        assert meta["question_id"] == "q1"
        assert meta["config_name"] == "cfg"
        assert meta["evaluation_type"] == "end_to_end"
        assert "abstained" in meta
        assert "abstention_level" in meta
        assert "confidence" in meta
        assert "retrieval_chunk_ids" in meta
        assert "generator_model" in meta

    def test_expected_abstention_preserved(self):
        r = _result(expected_abstention=True)
        items = prepare_e2e_items([r], "cfg")
        assert items[0].expected_abstention is True

    def test_category_preserved(self):
        r = _result(category="multi_source")
        items = prepare_e2e_items([r], "cfg")
        assert items[0].category == "multi_source"

    def test_empty_results_returns_empty_list(self):
        items = prepare_e2e_items([], "cfg")
        assert items == []
