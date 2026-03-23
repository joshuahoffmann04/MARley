"""Tests for LLM output abstention detection."""

from __future__ import annotations

import pytest

from src.marley.abstention.detection import (
    ABSTENTION_PREFIX,
    detect_abstention,
    extract_abstention_reason,
)


class TestDetectAbstention:
    """Tests for the detect_abstention function."""

    def test_exact_prefix(self) -> None:
        assert detect_abstention("ABSTENTION: not enough info") is True

    def test_case_insensitive(self) -> None:
        assert detect_abstention("abstention: not enough info") is True
        assert detect_abstention("Abstention: some reason") is True

    def test_leading_whitespace(self) -> None:
        assert detect_abstention("  ABSTENTION: reason") is True
        assert detect_abstention("\nABSTENTION: reason") is True

    def test_non_abstention_answer(self) -> None:
        assert detect_abstention("The study period is four semesters.") is False

    def test_empty_string(self) -> None:
        assert detect_abstention("") is False

    def test_no_false_positive_partial_match(self) -> None:
        """The word abstention in the middle should not trigger."""
        assert detect_abstention("There is no abstention needed here.") is False

    def test_multiline_response(self) -> None:
        assert detect_abstention("ABSTENTION: reason\nMore text") is True

    def test_prefix_without_reason(self) -> None:
        assert detect_abstention("ABSTENTION:") is True


class TestExtractAbstentionReason:
    """Tests for the extract_abstention_reason function."""

    def test_extracts_reason_text(self) -> None:
        reason = extract_abstention_reason("ABSTENTION: context lacks information about credits")
        assert reason == "context lacks information about credits"

    def test_strips_whitespace(self) -> None:
        reason = extract_abstention_reason("ABSTENTION:   extra spaces  ")
        assert reason == "extra spaces"

    def test_non_abstention_returns_empty(self) -> None:
        assert extract_abstention_reason("Normal answer text.") == ""

    def test_empty_reason(self) -> None:
        assert extract_abstention_reason("ABSTENTION:") == ""
