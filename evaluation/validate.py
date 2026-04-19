"""Data requirement validation for evaluation steps.

Checks that all required files and services are available before
running evaluation steps, providing clear actionable error messages.

Importable as a library (:func:`validate_data_requirements`) or
runnable as a CLI::

    python -m evaluation.validate --steps retrieval generation abstention e2e
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.marley.server.config import CHUNK_PATHS, check_ollama

# Evaluation file paths (relative to project root)
EVAL_PATHS: dict[str, str] = {
    "stpo": "data/evaluation/evaluation-stpo.json",
    "faq-stpo": "data/evaluation/evaluation-faq-stpo.json",
    "faq-ao": "data/evaluation/evaluation-faq-ao.json",
}

# Steps that require Ollama
OLLAMA_STEPS: set[str] = {"generation", "abstention", "e2e"}


def validate_data_requirements(
    steps: list[str],
    output_dir: str | Path = "data/evaluation",
    ollama_url: str = "http://localhost:11434",
) -> list[str]:
    """Check that all required files exist for the given evaluation steps.

    Args:
        steps: List of evaluation step names to validate.
            Valid steps: "retrieval", "rrf-tuning", "generation",
            "abstention", "e2e".
        output_dir: Base directory for evaluation output.
        ollama_url: Ollama server URL (checked for steps that need it).

    Returns:
        List of error messages (empty = all OK).
    """
    errors: list[str] = []

    # All steps need chunk files and evaluation files
    for kb, rel_path in CHUNK_PATHS.items():
        if not Path(rel_path).exists():
            errors.append(
                f"Missing chunk file: {rel_path}\n"
                f"  -> Run extraction and chunking for '{kb}' first.",
            )

    for kb, rel_path in EVAL_PATHS.items():
        if not Path(rel_path).exists():
            errors.append(
                f"Missing evaluation file: {rel_path}\n"
                f"  -> Create evaluation dataset for '{kb}'.",
            )

    # Check Ollama for steps that need it
    needs_ollama = any(s in OLLAMA_STEPS for s in steps)
    if needs_ollama:
        status = check_ollama(ollama_url)
        if not status["available"]:
            step_names = ", ".join(s for s in steps if s in OLLAMA_STEPS)
            errors.append(
                f"Ollama not available at {ollama_url}\n"
                f"  -> Start Ollama: ollama serve\n"
                f"  -> Required for: {step_names}",
            )

    return errors


def _cli(argv: list[str] | None = None) -> int:
    """CLI wrapper: run validation and exit non-zero on any error."""
    parser = argparse.ArgumentParser(
        description="Validate MARley evaluation prerequisites.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["retrieval", "rrf-tuning", "generation", "abstention", "e2e"],
        help="Evaluation steps to validate (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation",
        help="Base directory for evaluation output.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (checked for steps that need it).",
    )
    args = parser.parse_args(argv)

    errors = validate_data_requirements(
        args.steps,
        output_dir=args.output_dir,
        ollama_url=args.ollama_url,
    )
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        return 1
    print("All evaluation prerequisites satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
