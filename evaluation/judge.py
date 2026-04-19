"""RAGAS judge factory for the MARley generation evaluation.

The generator under evaluation is always the Ollama-served chat model.
The *judge* that scores its answers is configurable:

* ``"ollama"`` routes RAGAS to the local Ollama server (OpenAI-compatible API).
* ``"openai"`` routes RAGAS to the OpenAI API using ``gpt-4o-mini``.

Embeddings are fixed to ``sentence-transformers/all-mpnet-base-v2`` on CUDA
for both backends — retrieval-quality parity and GPU throughput matter more
than per-judge embedding variation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

JudgeBackend = Literal["ollama", "openai"]

_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
_OLLAMA_BATCH_SIZE = 20
"""Ollama batch size: OLLAMA_NUM_PARALLEL=2 × 10 waves keeps queues short
and avoids RAGAS retry storms from slow sequential generation."""

_OPENAI_BATCH_SIZE = 50
"""OpenAI batch size: rate-limit-bound, not concurrency-bound. RAGAS's
async client handles higher parallelism cleanly under tier-1 quotas."""

_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class Judge:
    """Composed judge configuration produced by :func:`make_judge`.

    Attributes
    ----------
    llm:
        A RAGAS-compatible LLM wrapper (from ``ragas.llms.llm_factory``)
        bound to either the Ollama server or the OpenAI API.
    embeddings:
        A RAGAS-compatible embeddings object running on CUDA. Shared across
        metrics that need semantic similarity (e.g. Answer Relevancy).
    batch_size:
        Samples per ``batch_score()`` call. Tuned per backend so the slow
        path (Ollama) doesn't blow up and the fast path (OpenAI) doesn't
        waste throughput.
    """

    llm: Any
    embeddings: Any
    batch_size: int


def make_judge(
    backend: JudgeBackend,
    *,
    ollama_model: str = "llama3.1:latest",
    ollama_url: str = "http://localhost:11434",
    ollama_judge_model: str | None = None,
) -> Judge:
    """Build a :class:`Judge` for the requested backend.

    Parameters
    ----------
    backend:
        ``"ollama"`` (default pipeline) or ``"openai"`` (faster, external).
    ollama_model:
        The Ollama model used by the *generator* (the system being
        evaluated). Ignored by the OpenAI judge. Used as the default
        judge model when ``ollama_judge_model`` is not provided.
    ollama_url:
        Ollama server URL. Ignored by the OpenAI judge.
    ollama_judge_model:
        Optional: when set and ``backend == "ollama"``, the judge runs
        on this model instead of ``ollama_model``. The intended use is
        to pair a small generator (e.g. ``llama3.1:latest``) with a
        larger, more structured-output-capable judge (e.g.
        ``qwen2.5:14b``). Must be pulled on the Ollama server before
        the run. Ignored when ``backend == "openai"``.

    Raises
    ------
    RuntimeError
        If ``backend == "openai"`` and ``OPENAI_API_KEY`` is not present
        in the environment. The caller is expected to load ``.env`` first.
    ValueError
        If ``backend`` is neither ``"ollama"`` nor ``"openai"``.
    """
    if backend not in ("ollama", "openai"):
        raise ValueError(
            f"Unknown judge backend: {backend!r}. Expected 'ollama' or 'openai'."
        )

    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    embeddings = HuggingFaceEmbeddings(
        model=_EMBEDDING_MODEL,
        device="cuda",
    )

    if backend == "ollama":
        judge_model = ollama_judge_model or ollama_model
        client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")
        llm = llm_factory(judge_model, provider="openai", client=client)
        return Judge(llm=llm, embeddings=embeddings, batch_size=_OLLAMA_BATCH_SIZE)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or use --judge ollama."
        )
    client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory(_OPENAI_MODEL, provider="openai", client=client)
    return Judge(llm=llm, embeddings=embeddings, batch_size=_OPENAI_BATCH_SIZE)
