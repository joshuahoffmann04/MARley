"""Prompt templates for the MARley generator.

Defines the system prompt and context formatting used to instruct the LLM
to answer student questions based solely on provided context chunks.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a study advisor for the M.Sc. Computer Science program "
    "at Philipps-Universität Marburg. Answer the student's question "
    "based ONLY on the provided context. Be concise and precise. "
    "If the context does not contain enough information to answer "
    "the question, state that clearly."
)


def format_context(chunks: list[dict]) -> str:
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


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
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
