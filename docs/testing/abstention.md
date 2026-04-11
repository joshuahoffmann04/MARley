# Abstention Tests

> Tests for the two-level abstention detection in `src/marley/abstention/`.

**Test file**: `tests/abstention/test_detection.py`
**Total**: 12 tests

## Overview

The abstention module detects when the LLM signals that it cannot answer a query.
Level-2 abstention detection works by checking whether the LLM output starts with
the `ABSTENTION:` prefix. Level-1 (retrieval confidence) is tested as part of the
pipeline tests in [server.md](server.md).

## TestDetectAbstention (8 tests)

Tests for the `detect_abstention()` function.

| Test | Validates |
|------|-----------|
| `test_exact_prefix` | `"ABSTENTION: not enough info"` is detected |
| `test_case_insensitive` | `"abstention:"` and `"Abstention:"` both detected |
| `test_leading_whitespace` | Leading spaces and newlines are ignored |
| `test_non_abstention_answer` | Normal answer text returns `False` |
| `test_empty_string` | Empty string returns `False` |
| `test_no_false_positive_partial_match` | "abstention" mid-sentence is not a match |
| `test_multiline_response` | First line with prefix is detected even with more text |
| `test_prefix_without_reason` | `"ABSTENTION:"` alone (no reason) is detected |

## TestExtractAbstentionReason (4 tests)

Tests for the `extract_abstention_reason()` function.

| Test | Validates |
|------|-----------|
| `test_extracts_reason_text` | Extracts text after `"ABSTENTION: "` |
| `test_strips_whitespace` | Extra whitespace in reason is stripped |
| `test_non_abstention_returns_empty` | Non-abstention text returns `""` |
| `test_empty_reason` | `"ABSTENTION:"` with no reason returns `""` |
