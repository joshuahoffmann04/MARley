"""Prompt templates for the MARley LLM judge.

Contains the system prompt and user message formatter used to instruct
the judge model to score generated answers on faithfulness, answer
relevance, and correctness.

The judge uses a single structured prompt that requests all three scores
in one call, reducing API latency and token usage compared to three
separate evaluation calls.
"""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge for a Retrieval-Augmented Generation (RAG) "
    "question-answering system.\n\n"
    "Your task is to evaluate a generated answer on three criteria.\n"
    "You MUST respond with a JSON object containing exactly these three fields:\n\n"
    '{"faithfulness": <float 0.0-1.0>, '
    '"answer_relevance": <float 0.0-1.0>, '
    '"correctness": <float 0.0-1.0>}\n\n'
    "Criteria definitions:\n\n"
    "  faithfulness: Does the generated answer ONLY use information present in the "
    "provided context? Score 1.0 if every claim is grounded in the context. "
    "Score 0.0 if the answer contains information not present in the context "
    "(hallucination). Partial grounding yields intermediate scores.\n\n"
    "  answer_relevance: Does the generated answer address the question that was "
    "asked? Score 1.0 if the answer fully answers the question. "
    "Score 0.0 if the answer is completely off-topic or empty.\n\n"
    "  correctness: Does the generated answer agree with the reference answer? "
    "Score 1.0 if the answer is factually correct and complete relative to the "
    "reference. Score 0.0 if the answer contradicts or ignores the reference.\n\n"
    "IMPORTANT: Output ONLY the JSON object. Do not add any explanation."
)


def format_context(chunks: list[dict]) -> str:
    """Format context chunks for the judge prompt.

    Args:
        chunks: List of chunk dicts with at least a 'text' key.

    Returns:
        Numbered context string, or 'No context provided.' if empty.
    """
    if not chunks:
        return "No context provided."
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))


def build_judge_messages(
    question: str,
    context: list[dict],
    generated_answer: str,
    reference_answer: str,
) -> list[dict]:
    """Build the message list for a judge LLM call.

    Returns a two-message list (system + user) following the standard
    chat format.

    Args:
        question: The original question text.
        context: Retrieval context chunks.
        generated_answer: The answer to evaluate.
        reference_answer: The ground-truth reference answer.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    context_str = format_context(context)
    user_content = (
        f"Question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Generated Answer: {generated_answer}\n\n"
        f"Reference Answer: {reference_answer}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
