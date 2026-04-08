"""Tests for the judge prompts module."""

from __future__ import annotations

from evaluation.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_messages,
    format_context,
)


class TestFormatContext:
    def test_empty_returns_fallback(self):
        assert format_context([]) == "No context provided."

    def test_single_chunk(self):
        chunks = [{"chunk_id": "c1", "text": "The study period is 4 semesters."}]
        result = format_context(chunks)
        assert "[1]" in result
        assert "4 semesters" in result

    def test_multiple_chunks_numbered(self):
        chunks = [
            {"chunk_id": "c1", "text": "First passage."},
            {"chunk_id": "c2", "text": "Second passage."},
        ]
        result = format_context(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "First passage." in result
        assert "Second passage." in result

    def test_chunks_in_order(self):
        chunks = [
            {"chunk_id": "c1", "text": "First."},
            {"chunk_id": "c2", "text": "Second."},
        ]
        result = format_context(chunks)
        assert result.index("First.") < result.index("Second.")


class TestBuildJudgeMessages:
    def test_returns_two_messages(self):
        msgs = build_judge_messages("Q?", [], "A.", "Ref.")
        assert len(msgs) == 2

    def test_system_message_first(self):
        msgs = build_judge_messages("Q?", [], "A.", "Ref.")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == JUDGE_SYSTEM_PROMPT

    def test_user_message_contains_question(self):
        msgs = build_judge_messages("How long?", [], "A.", "Ref.")
        assert "How long?" in msgs[1]["content"]

    def test_user_message_contains_generated_answer(self):
        msgs = build_judge_messages("Q?", [], "My generated answer.", "Ref.")
        assert "My generated answer." in msgs[1]["content"]

    def test_user_message_contains_reference_answer(self):
        msgs = build_judge_messages("Q?", [], "A.", "The reference answer.")
        assert "The reference answer." in msgs[1]["content"]

    def test_user_message_contains_context(self):
        chunks = [{"chunk_id": "c1", "text": "Context passage."}]
        msgs = build_judge_messages("Q?", chunks, "A.", "Ref.")
        assert "Context passage." in msgs[1]["content"]

    def test_empty_context_in_message(self):
        msgs = build_judge_messages("Q?", [], "A.", "Ref.")
        assert "No context provided." in msgs[1]["content"]


class TestJudgeSystemPrompt:
    def test_contains_faithfulness(self):
        assert "faithfulness" in JUDGE_SYSTEM_PROMPT.lower()

    def test_contains_answer_relevance(self):
        assert "answer_relevance" in JUDGE_SYSTEM_PROMPT

    def test_contains_correctness(self):
        assert "correctness" in JUDGE_SYSTEM_PROMPT.lower()

    def test_contains_json_instruction(self):
        assert "JSON" in JUDGE_SYSTEM_PROMPT
