"""Tests for the judge base module (JudgementResult + Judge ABC)."""

from __future__ import annotations

import pytest

from evaluation.judge.base import Judge, JudgementResult


# ---------------------------------------------------------------------------
# Concrete stub for testing the ABC
# ---------------------------------------------------------------------------

class _FixedJudge(Judge):
    """Judge stub that returns fixed scores regardless of input."""

    def __init__(self, f: float = 0.9, r: float = 0.8, c: float = 0.7) -> None:
        self._f, self._r, self._c = f, r, c

    @property
    def model(self) -> str:
        return "stub-judge"

    def judge(
        self,
        question_id: str,
        question: str,
        context: list[dict],
        generated_answer: str,
        reference_answer: str,
    ) -> JudgementResult:
        return JudgementResult(
            question_id=question_id,
            faithfulness=self._f,
            answer_relevance=self._r,
            correctness=self._c,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# TestJudgementResult
# ---------------------------------------------------------------------------


class TestJudgementResult:
    def test_fields_accessible(self):
        result = JudgementResult(
            question_id="eval-001",
            faithfulness=0.9,
            answer_relevance=0.8,
            correctness=0.7,
            model="stub",
        )
        assert result.question_id == "eval-001"
        assert result.faithfulness == pytest.approx(0.9)
        assert result.answer_relevance == pytest.approx(0.8)
        assert result.correctness == pytest.approx(0.7)
        assert result.model == "stub"

    def test_is_dataclass(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(JudgementResult)}
        assert field_names == {
            "question_id",
            "faithfulness",
            "answer_relevance",
            "correctness",
            "model",
        }


# ---------------------------------------------------------------------------
# TestJudgeABC
# ---------------------------------------------------------------------------


class TestJudgeABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            Judge()  # type: ignore[abstract]

    def test_stub_implements_interface(self):
        judge = _FixedJudge()
        assert isinstance(judge, Judge)

    def test_model_property(self):
        judge = _FixedJudge()
        assert judge.model == "stub-judge"

    def test_model_is_property(self):
        assert isinstance(type(_FixedJudge()).model, property)

    def test_judge_returns_judgement_result(self):
        judge = _FixedJudge()
        result = judge.judge(
            question_id="q1",
            question="How long?",
            context=[{"chunk_id": "c1", "text": "4 semesters."}],
            generated_answer="4 semesters.",
            reference_answer="4 semesters.",
        )
        assert isinstance(result, JudgementResult)

    def test_judge_propagates_question_id(self):
        judge = _FixedJudge()
        result = judge.judge(
            question_id="eval-042",
            question="q",
            context=[],
            generated_answer="a",
            reference_answer="a",
        )
        assert result.question_id == "eval-042"

    def test_fixed_scores(self):
        judge = _FixedJudge(f=1.0, r=0.5, c=0.0)
        result = judge.judge("q1", "q", [], "a", "a")
        assert result.faithfulness == pytest.approx(1.0)
        assert result.answer_relevance == pytest.approx(0.5)
        assert result.correctness == pytest.approx(0.0)
