"""Prompt templates for the MARley generator.

Defines the system prompt and context formatting used to instruct the LLM
to answer student questions based solely on provided context chunks.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a study advisor for the M.Sc. Computer Science program "
    "at Philipps-Universit\u00e4t Marburg.\n\n"
    "Answer the student's question using ONLY the numbered context "
    "passages below. Follow these rules:\n\n"
    "1. Base your answer exclusively on information from the provided "
    "context passages.\n"
    "2. Be concise, precise, and factually accurate.\n"
    "3. If the provided context does not contain sufficient information "
    "to fully answer the question, respond with exactly:\n"
    "   ABSTENTION: <reason>\n"
    "   In particular, if the context is silent on the question - i.e. "
    "the topic is simply not addressed - you MUST abstain. Do NOT "
    "generalise from the absence of a prohibition to a 'Yes', and do "
    "NOT generalise from the absence of a permission to a 'No'.\n"
    "4. Never guess, speculate, or supplement with knowledge not present "
    "in the context.\n"
    "5. Write your answer as plain text for the student. Do NOT reference "
    "context passage numbers (e.g. [1], [2]), chunk IDs, or source labels "
    "in your answer. The student does not see the context passages."
)


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format a list of chunk dicts into a numbered context string.

    Each chunk dict must have a 'text' key. The chunks are numbered
    starting from 1 to help the LLM reference specific passages.
    """
    if not chunks:
        return "No context provided."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['text']}")
    return "\n\n".join(parts)


def build_messages(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the message list for an LLM chat call.

    Returns a list of message dicts with 'role' and 'content' keys,
    following the standard chat format (system, user).
    """
    context_str = format_context(chunks)
    user_content = f"Context:\n{context_str}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
