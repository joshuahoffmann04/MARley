"""Tests for the generation evaluation runner.

Uses stub Generator and Judge implementations to test the evaluation
pipeline without requiring a running Ollama server.
"""

from __future__ import annotations

import pytest

from evaluation.generation.evaluate import (
    _assemble_context,
    load_evaluation,
    run_generation_evaluation,
    select_distractors,
)
from evaluation.generation.judge import Judge, JudgementResult
from evaluation.generation.metrics import GenerationEvalResult
from src.marley.generator.base import Generator
from src.marley.models.generation import GenerationResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubGenerator(Generator):
    """Returns a fixed answer for any query."""

    def __init__(self, answer: str = "stub answer"):
        self.answer = answer
        self.model = "stub-model"

    def generate(self, query: str, context: list[dict]) -> GenerationResult:
        return GenerationResult(
            answer=self.answer,
            model=self.model,
            context_chunk_ids=[c["chunk_id"] for c in context if "chunk_id" in c],
            prompt_tokens=10,
            completion_tokens=5,
        )


class StubJudge(Judge):
    """Always judges answers as correct."""

    def __init__(self, correct: bool = True):
        self._correct = correct
        self.model = "stub-judge"

    def evaluate(self, question, reference_answer, generated_answer):
        return JudgementResult(
            correct=self._correct,
            confidence=1.0,
            reasoning="stub",
        )


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

CORPUS = [
    {"chunk_id": "c1", "text": "The study period is 4 semesters.", "metadata": {}},
    {"chunk_id": "c2", "text": "The thesis has 30 credits.", "metadata": {}},
    {"chunk_id": "c3", "text": "Examination rules apply.", "metadata": {}},
    {"chunk_id": "c4", "text": "Study abroad is possible.", "metadata": {}},
    {"chunk_id": "c5", "text": "Module handbook details.", "metadata": {}},
]

QUESTIONS = [
    {
        "id": "eval-001",
        "question": "How long is the study period?",
        "reference_answer": "4 semesters.",
        "category": "direct",
        "relevant_chunks": ["c1"],
        "expected_abstention": False,
    },
    {
        "id": "eval-002",
        "question": "How many credits for the thesis?",
        "reference_answer": "30 credits.",
        "category": "direct",
        "relevant_chunks": ["c2"],
        "expected_abstention": False,
    },
    {
        "id": "eval-076",
        "question": "What is the tuition fee?",
        "reference_answer": "",
        "category": "unanswerable",
        "relevant_chunks": [],
        "expected_abstention": True,
    },
]


# ---------------------------------------------------------------------------
# TestSelectDistractors
# ---------------------------------------------------------------------------


class TestSelectDistractors:
    """Tests for select_distractors()."""

    def test_excludes_relevant_chunks(self):
        distractors = select_distractors("study period", {"c1"}, CORPUS, 3)
        ids = {d["chunk_id"] for d in distractors}
        assert "c1" not in ids

    def test_returns_up_to_requested_count(self):
        distractors = select_distractors("study period", {"c1"}, CORPUS, 2)
        assert len(distractors) <= 2
        assert len(distractors) >= 1

    def test_returns_empty_for_zero(self):
        distractors = select_distractors("study period", {"c1"}, CORPUS, 0)
        assert distractors == []

    def test_returns_at_most_available(self):
        # Only 4 non-relevant chunks, but requesting 10
        distractors = select_distractors("study period", {"c1"}, CORPUS, 10)
        assert len(distractors) <= 4

    def test_deterministic(self):
        d1 = select_distractors("study period", {"c1"}, CORPUS, 3)
        d2 = select_distractors("study period", {"c1"}, CORPUS, 3)
        assert [d["chunk_id"] for d in d1] == [d["chunk_id"] for d in d2]


# ---------------------------------------------------------------------------
# TestAssembleContext
# ---------------------------------------------------------------------------


class TestAssembleContext:
    """Tests for _assemble_context()."""

    def test_includes_relevant_and_distractors(self):
        relevant = [CORPUS[0]]
        distractors = [CORPUS[2], CORPUS[3]]
        ctx = _assemble_context(relevant, distractors, 2, seed=42)
        ids = {c["chunk_id"] for c in ctx}
        assert ids == {"c1", "c3", "c4"}

    def test_limits_distractors(self):
        relevant = [CORPUS[0]]
        distractors = [CORPUS[2], CORPUS[3], CORPUS[4]]
        ctx = _assemble_context(relevant, distractors, 1, seed=42)
        assert len(ctx) == 2  # 1 relevant + 1 distractor

    def test_deterministic_with_same_seed(self):
        relevant = [CORPUS[0]]
        distractors = [CORPUS[2], CORPUS[3]]
        ctx1 = _assemble_context(relevant, distractors, 2, seed=42)
        ctx2 = _assemble_context(relevant, distractors, 2, seed=42)
        assert [c["chunk_id"] for c in ctx1] == [c["chunk_id"] for c in ctx2]

    def test_zero_distractors(self):
        relevant = [CORPUS[0]]
        ctx = _assemble_context(relevant, [CORPUS[2]], 0, seed=42)
        assert len(ctx) == 1
        assert ctx[0]["chunk_id"] == "c1"


# ---------------------------------------------------------------------------
# TestRunGenerationEvaluation
# ---------------------------------------------------------------------------


class TestRunGenerationEvaluation:
    """Tests for run_generation_evaluation()."""

    def test_skips_unanswerable(self):
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=[0],
        )
        question_ids = {r.question_id for r in results}
        assert "eval-076" not in question_ids

    def test_evaluates_answerable_questions(self):
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=[0],
        )
        question_ids = {r.question_id for r in results}
        assert "eval-001" in question_ids
        assert "eval-002" in question_ids

    def test_all_distractor_levels(self):
        levels = [0, 1, 2]
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=levels,
        )
        # 2 answerable questions × 3 levels = 6 results
        assert len(results) == 6

    def test_result_type(self):
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=[0],
        )
        assert all(isinstance(r, GenerationEvalResult) for r in results)

    def test_records_generated_answer(self):
        results = run_generation_evaluation(
            StubGenerator(answer="test answer"), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=[0],
        )
        assert all(r.generated_answer == "test answer" for r in results)

    def test_records_judge_result(self):
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(correct=False), CORPUS, QUESTIONS,
            distractor_levels=[0],
        )
        assert all(r.correct is False for r in results)

    def test_context_grows_with_distractors(self):
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=[0, 3],
        )
        for r in results:
            if r.num_distractors == 0:
                # Only relevant chunks
                assert len(r.context_chunk_ids) == 1
            elif r.num_distractors == 3:
                # Relevant + up to 3 distractors
                assert len(r.context_chunk_ids) >= 2

    def test_skips_questions_without_relevant_chunks(self):
        questions = [
            {
                "id": "eval-099",
                "question": "Test?",
                "reference_answer": "Answer.",
                "relevant_chunks": [],
                "expected_abstention": False,
            },
        ]
        results = run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, questions,
            distractor_levels=[0],
        )
        assert len(results) == 0

    def test_progress_callback_called(self):
        calls = []
        run_generation_evaluation(
            StubGenerator(), StubJudge(), CORPUS, QUESTIONS,
            distractor_levels=[0],
            progress_callback=lambda qid, n: calls.append((qid, n)),
        )
        assert len(calls) == 2  # 2 answerable questions
