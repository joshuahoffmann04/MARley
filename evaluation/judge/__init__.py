"""LLM judge module for the MARley evaluation framework.

Provides the Judge abstract base class and two concrete implementations:
- OllamaJudge: uses a locally hosted Ollama model (default, no API key needed)
- OpenAIJudge: uses the OpenAI API (requires OPENAI_API_KEY)

Both judges evaluate generated answers on three criteria:
  faithfulness    — answer only uses information from the retrieved context
  answer_relevance — answer addresses the question asked
  correctness     — answer agrees with the reference answer
"""

from evaluation.judge.base import Judge, JudgementResult
from evaluation.judge.ollama_judge import OllamaJudge
from evaluation.judge.openai_judge import OpenAIJudge

__all__ = [
    "Judge",
    "JudgementResult",
    "OllamaJudge",
    "OpenAIJudge",
]
