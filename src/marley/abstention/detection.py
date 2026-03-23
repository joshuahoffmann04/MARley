"""LLM output abstention detection.

Detects whether a generator response is an explicit abstention using the
structured ABSTENTION: prefix format. This provides deterministic detection
without heuristic keyword matching.
"""

from __future__ import annotations

ABSTENTION_PREFIX = "ABSTENTION:"


def detect_abstention(answer: str) -> bool:
    """Detect whether the LLM output is an abstention.

    Returns True if the answer starts with the ABSTENTION_PREFIX
    (case-insensitive, ignoring leading whitespace).
    """
    return answer.strip().upper().startswith(ABSTENTION_PREFIX)


def extract_abstention_reason(answer: str) -> str:
    """Extract the reason from an abstention response.

    Returns the text after 'ABSTENTION:', stripped of leading/trailing
    whitespace. Returns an empty string if the answer is not an abstention.
    """
    stripped = answer.strip()
    if not stripped.upper().startswith(ABSTENTION_PREFIX):
        return ""
    return stripped[len(ABSTENTION_PREFIX):].strip()
