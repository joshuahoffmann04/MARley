"""Ollama-based generator for the MARley pipeline.

Uses a locally hosted Ollama model for answer generation. The model
and server URL are configurable via constructor parameters.
"""

from __future__ import annotations

import ollama as ollama_lib

from src.marley.generator.base import Generator
from src.marley.generator.prompt import build_messages
from src.marley.models.generation import GenerationResult


class OllamaGenerator(Generator):
    """Generate answers using an Ollama-hosted LLM."""

    def __init__(
        self,
        model: str = "llama3.1:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._client = ollama_lib.Client(host=base_url)

    def generate(self, query: str, context: list[dict]) -> GenerationResult:
        """Generate an answer for the query using the provided context.

        Sends a chat request to the Ollama server and returns a
        structured GenerationResult.
        """
        messages = build_messages(query, context)
        chunk_ids = [c["chunk_id"] for c in context if "chunk_id" in c]

        response = self._client.chat(model=self.model, messages=messages)

        return GenerationResult(
            answer=response.message.content.strip(),
            model=response.model or self.model,
            context_chunk_ids=chunk_ids,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
        )
