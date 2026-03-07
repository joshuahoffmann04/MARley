"""Generation strategies for the MARley pipeline."""

from src.marley.generator.base import Generator
from src.marley.generator.ollama import OllamaGenerator

__all__ = [
    "Generator",
    "OllamaGenerator",
]
