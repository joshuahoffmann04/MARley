"""Run all 33 end-to-end evaluation configurations.

Iterates over every configuration from generate_all_configs(),
builds the retriever, runs the full pipeline for 100 questions,
and saves the raw results with automatic abstention metrics.

Supports resuming: skips configs whose output files already exist.

Usage::

    python -m evaluation.end_to_end.run_all
    python -m evaluation.end_to_end.run_all --output-dir data/testing
    python -m evaluation.end_to_end.run_all --config-filter "single-stpo"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from evaluation.end_to_end.config import (
    KNOWLEDGE_BASES,
    E2EConfig,
    generate_all_configs,
)
from evaluation.end_to_end.evaluate import (
    load_questions,
    run_e2e_config,
    save_report,
    sweep_threshold,
)
from src.marley.generator.ollama import OllamaGenerator
from src.marley.models.retrieval import Retriever
from src.marley.retrieval import (
    BM25Retriever,
    FusionRetriever,
    HybridRetriever,
    VectorRetriever,
    load_chunks,
)
from src.marley.server.config import CHUNK_PATHS, check_ollama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever construction
# ---------------------------------------------------------------------------


def _vector_persist_dir(output_dir: Path, tag: str) -> str:
    """Return a unique ChromaDB persist directory."""
    return str(output_dir / ".chromadb-e2e" / tag)


def _create_retriever(
    retriever_type: str,
    tag: str,
    output_dir: Path,
    k_rrf_hybrid: int = 60,
) -> Retriever:
    """Create a new (unindexed) retriever instance."""
    if retriever_type == "bm25":
        return BM25Retriever()
    if retriever_type == "vector":
        return VectorRetriever(
            persist_directory=_vector_persist_dir(output_dir, f"vector-{tag}"),
        )
    if retriever_type == "hybrid":
        return HybridRetriever((
            BM25Retriever(),
            VectorRetriever(
                persist_directory=_vector_persist_dir(output_dir, f"hybrid-{tag}"),
            ),
        ), k_rrf=k_rrf_hybrid)
    raise ValueError(f"Unknown retriever type: {retriever_type}")


def build_retriever(
    config: E2EConfig,
    output_dir: Path,
) -> Retriever:
    """Build and index a retriever for the given E2E config."""
    kbs = list(config.knowledge_bases)

    if config.strategy == "fusion":
        sub_retrievers: list[Retriever] = []
        for kb in kbs:
            r = _create_retriever(config.retriever_type, f"fusion-{kb}", output_dir, k_rrf_hybrid=config.k_rrf_hybrid)
            chunks = load_chunks(CHUNK_PATHS[kb])
            r.index(chunks)
            sub_retrievers.append(r)
        return FusionRetriever(sub_retrievers, k_rrf=config.k_rrf)

    # single or merged_pool
    kb_tag = "-".join(sorted(kbs))
    r = _create_retriever(config.retriever_type, f"{config.strategy}-{kb_tag}", output_dir, k_rrf_hybrid=config.k_rrf_hybrid)
    all_chunks: list[dict] = []
    for kb in kbs:
        all_chunks.extend(load_chunks(CHUNK_PATHS[kb]))
    r.index(all_chunks)
    return r


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all(
    output_dir: Path,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.1:latest",
    config_filter: str | None = None,
) -> None:
    """Run all E2E configurations and save results.

    Args:
        output_dir: Directory for output files.
        ollama_url: Ollama server URL.
        ollama_model: Model name for generation.
        config_filter: If set, only run configs whose name contains this substring.
    """
    # Check Ollama
    ollama_status = check_ollama(ollama_url)
    if not ollama_status["available"]:
        logger.error(
            "Ollama not reachable at %s: %s",
            ollama_url, ollama_status.get("error", "unknown"),
        )
        logger.error("Start Ollama first, then re-run.")
        sys.exit(1)

    logger.info("Ollama connected at %s (model: %s)", ollama_url, ollama_model)

    # Load questions from master evaluation dataset
    eval_path = Path("data/testing/evaluation.json")
    if not eval_path.exists():
        logger.error("Evaluation dataset not found: %s", eval_path)
        sys.exit(1)

    questions = load_questions(eval_path)
    logger.info("Loaded %d evaluation questions", len(questions))

    # Initialize generator
    generator = OllamaGenerator(model=ollama_model, base_url=ollama_url)

    # Generate all configs
    all_configs = generate_all_configs()
    if config_filter:
        all_configs = [c for c in all_configs if config_filter in c.name]
        logger.info("Filtered to %d configs matching '%s'", len(all_configs), config_filter)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_configs = len(all_configs)
    completed = 0
    skipped = 0

    for idx, config in enumerate(all_configs, 1):
        report_path = output_dir / f"e2e-results-{config.name}.json"

        # Resume support: skip if output already exists
        if report_path.exists():
            logger.info(
                "[%d/%d] SKIP %s (output exists)", idx, total_configs, config.name,
            )
            skipped += 1
            continue

        logger.info(
            "[%d/%d] START %s (type=%s, kbs=%s, strategy=%s)",
            idx, total_configs, config.name,
            config.retriever_type, list(config.knowledge_bases), config.strategy,
        )
        start_time = time.time()

        # Build retriever
        retriever = build_retriever(config, output_dir)
        logger.info("  Retriever built (%d docs indexed)", retriever.size)

        # Determine normalization params
        norm_params: dict = {}
        if config.normalization_strategy == "rrf":
            if config.strategy == "fusion":
                n_retrievers = len(config.knowledge_bases)
            else:
                n_retrievers = 2  # hybrid has 2 sub-retrievers
            norm_params = {"rrf_n_retrievers": n_retrievers, "rrf_k": config.k_rrf}

        # Step 1: Sweep threshold (no LLM)
        best_threshold, sweep = sweep_threshold(
            retriever, questions, config.normalization_strategy,
            k=config.k, normalization_params=norm_params,
        )
        logger.info("  Optimal threshold: %.2f", best_threshold)

        # Step 2: Run full pipeline
        def progress(current: int, total: int) -> None:
            if current % 10 == 0 or current == total:
                logger.info("  Progress: %d/%d questions", current, total)

        results = run_e2e_config(
            config, retriever, generator, questions,
            threshold=best_threshold,
            normalization_params=norm_params,
            progress_callback=progress,
        )

        # Compute abstention summary
        abstained_count = sum(1 for r in results if r.abstained)
        elapsed = time.time() - start_time
        logger.info(
            "  Done: %d questions, %d abstentions, %.1fs",
            len(results), abstained_count, elapsed,
        )

        # Step 3: Save report
        from evaluation.utils import compute_abstention_metrics
        abstention_input = [
            {"expected_abstention": r.expected_abstention, "system_abstained": r.abstained}
            for r in results
        ]
        abstention_metrics = compute_abstention_metrics(abstention_input, best_threshold)

        report = {
            "config": asdict(config),
            "threshold": best_threshold,
            "level1_sweep": sweep,
            "abstention_metrics": asdict(abstention_metrics),
            "generator_model": ollama_model,
            "results": [asdict(r) for r in results],
        }
        save_report(report, report_path)
        logger.info("  Report saved: %s", report_path)

        completed += 1

    # Summary
    logger.info("=" * 60)
    logger.info(
        "E2E EVALUATION COMPLETE: %d/%d configs run, %d skipped",
        completed, total_configs, skipped,
    )
    logger.info("Output directory: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all MARley end-to-end evaluation configurations.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/testing",
        help="Directory for output files (default: data/testing)",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model", type=str, default="llama3.1:latest",
        help="Ollama model name (default: llama3.1:latest)",
    )
    parser.add_argument(
        "--config-filter", type=str, default=None,
        help="Only run configs whose name contains this substring",
    )

    args = parser.parse_args()

    run_all(
        output_dir=Path(args.output_dir),
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        config_filter=args.config_filter,
    )


if __name__ == "__main__":
    main()
