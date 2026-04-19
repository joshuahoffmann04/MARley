"""Ollama-based generator for the MARley pipeline.

Uses a locally hosted Ollama model for answer generation. The model
and server URL are configurable via constructor parameters.
"""

from __future__ import annotations

from typing import Any

import ollama as ollama_lib

from src.marley.generator.base import Generator
from src.marley.generator.prompt import build_messages
from src.marley.models.generation import GenerationResult

# Generation defaults: low temperature for factual QA, and a bounded
# output length so judge payloads stay predictable without truncating
# legitimate enumerations (module lists, multi-point answers).
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_NUM_PREDICT: int = 512


class GenerationError(RuntimeError):
    """Raised when the Ollama server fails to produce a usable response."""


class OllamaGenerator(Generator):
    """Generate answers using an Ollama-hosted LLM."""

    def __init__(
        self,
        model: str = "llama3.1:latest",
        base_url: str = "http://localhost:11434",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ) -> None:
        self._model = model
        self._client = ollama_lib.Client(host=base_url)
        self._options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
        }

    @property
    def model(self) -> str:
        return self._model

    def generate(self, query: str, context: list[dict[str, Any]]) -> GenerationResult:
        """Generate an answer for the query using the provided context.

        Sends a chat request to the Ollama server and returns a
        structured :class:`GenerationResult`. Any underlying Ollama
        client, HTTP, or response-shape error is re-raised as a
        :class:`GenerationError` so the server layer can produce a
        generic 5xx response without leaking internals.
        """
        messages = build_messages(query, context)
        chunk_ids = [c["chunk_id"] for c in context if "chunk_id" in c]

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options=self._options,
            )
        except Exception as exc:  # network, timeout, JSON decode, etc.
            raise GenerationError(f"Ollama chat failed: {exc}") from exc

        try:
            content = response.message.content
        except AttributeError as exc:
            raise GenerationError("Ollama response missing message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Ollama returned an empty or non-string answer")

        return GenerationResult(
            answer=content.strip(),
            model=response.model or self.model,
            context_chunk_ids=chunk_ids,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
        )
