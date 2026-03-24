"""Unified evaluation CLI for the MARley pipeline.

Single entry point for all evaluation steps:

    python -m evaluation --check          # Validate data requirements
    python -m evaluation --retrieval      # Run retrieval evaluation
    python -m evaluation --rrf-tuning     # Sweep k_rrf for Hybrid and Fusion
    python -m evaluation --generation     # Run generation evaluation
    python -m evaluation --abstention     # Run abstention evaluation
    python -m evaluation --e2e            # Run end-to-end evaluation
    python -m evaluation --all            # Run everything in order

Execution order: retrieval -> rrf-tuning -> generation -> abstention -> e2e.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from evaluation.validate import EVAL_PATHS, validate_data_requirements
from src.marley.models.retrieval import Retriever
from src.marley.server.config import CHUNK_PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation")

# All steps in execution order
ALL_STEPS = ["retrieval", "rrf-tuning", "generation", "abstention", "e2e"]


def _save_json(data: dict | list, path: Path) -> None:
    """Save JSON output with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _create_retriever(
    retriever_type: str,
    persist_dir: str | None = None,
) -> "Retriever":
    """Create a retriever instance by type name.

    Args:
        retriever_type: One of ``"bm25"``, ``"vector"``, ``"hybrid"``.
        persist_dir: ChromaDB persist directory (required for vector/hybrid).
    """
    from src.marley.retrieval import BM25Retriever, HybridRetriever, VectorRetriever

    if retriever_type == "bm25":
        return BM25Retriever()
    if retriever_type == "vector":
        return VectorRetriever(persist_directory=persist_dir)
    if retriever_type == "hybrid":
        return HybridRetriever(
            (
                BM25Retriever(),
                VectorRetriever(persist_directory=persist_dir),
            )
        )
    raise ValueError(f"Unknown retriever type: {retriever_type}")


def _make_retriever_factory(
    retriever_type: str,
    persist_base: Path,
    prefix: str,
):
    """Return a factory callable producing fresh retrievers with unique paths.

    Each call to the returned factory creates a retriever with a unique
    ChromaDB persist directory, which is necessary for fusion evaluation
    where each KB gets its own retriever instance.
    """
    counter = [0]

    def factory():
        counter[0] += 1
        persist_dir = str(persist_base / f"{prefix}-{retriever_type}-{counter[0]}")
        return _create_retriever(retriever_type, persist_dir)

    return factory


_RETRIEVER_TYPES = ("bm25", "vector", "hybrid")


def run_retrieval_step(output_dir: Path) -> None:
    """Run retrieval evaluation: single-KB + combined (merged pool, fusion).

    Evaluates all three retriever types (BM25, Vector, Hybrid) across:
    - Single-KB evaluation (per KB)
    - Merged pool evaluation (all KBs combined)
    - Fusion evaluation (per-KB retrievers fused via RRF)
    """
    from evaluation.retrieval.combined import (
        run_fusion_evaluation,
        run_merged_pool_evaluation,
    )
    from evaluation.retrieval.evaluate import run_and_report
    from src.marley.retrieval import load_chunks

    logger.info("=" * 60)
    logger.info("RETRIEVAL EVALUATION")

    persist_base = output_dir / ".chromadb-retrieval"
    reports = []

    # Single-KB evaluation (all retriever types × all KBs)
    for retriever_type in _RETRIEVER_TYPES:
        for kb, chunk_path in CHUNK_PATHS.items():
            eval_path = EVAL_PATHS[kb]
            chunks = load_chunks(chunk_path)
            if not chunks:
                logger.info("  Skipping single-KB %s/%s (0 chunks)", retriever_type, kb)
                continue
            logger.info("  Single-KB: %s / %s", retriever_type, kb)
            persist_dir = str(persist_base / f"single-{retriever_type}-{kb}")
            retriever = _create_retriever(retriever_type, persist_dir)
            retriever.index(chunks)
            report = run_and_report(retriever, eval_path)
            report["knowledge_base"] = kb
            reports.append(report)

    # Combined: merged pool (all retriever types)
    for combo_name, combo_kbs in [
        ("stpo+faq-stpo+faq-ao", ["stpo", "faq-stpo", "faq-ao"]),
    ]:
        chunk_paths = {kb: CHUNK_PATHS[kb] for kb in combo_kbs}
        eval_paths = {kb: EVAL_PATHS[kb] for kb in combo_kbs}
        for retriever_type in _RETRIEVER_TYPES:
            logger.info("  Merged pool: %s / %s", retriever_type, combo_name)
            persist_dir = str(persist_base / f"merged-{retriever_type}")
            retriever = _create_retriever(retriever_type, persist_dir)
            report = run_merged_pool_evaluation(
                retriever,
                chunk_paths,
                eval_paths,
            )
            reports.append(report)

    # Combined: fusion (all retriever types)
    for combo_name, combo_kbs in [
        ("stpo+faq-stpo+faq-ao", ["stpo", "faq-stpo", "faq-ao"]),
    ]:
        chunk_paths = {kb: CHUNK_PATHS[kb] for kb in combo_kbs}
        eval_paths = {kb: EVAL_PATHS[kb] for kb in combo_kbs}
        for retriever_type in _RETRIEVER_TYPES:
            logger.info("  Fusion: %s / %s", retriever_type, combo_name)
            factory = _make_retriever_factory(
                retriever_type,
                persist_base,
                "fusion",
            )
            report = run_fusion_evaluation(
                retriever_factory=factory,
                chunk_paths=chunk_paths,
                eval_paths=eval_paths,
            )
            reports.append(report)

    _save_json(reports, output_dir / "retrieval-evaluation.json")


def run_rrf_tuning_step(output_dir: Path) -> None:
    """Run RRF k-parameter sweep for Hybrid and Fusion."""
    from evaluation.retrieval.rrf_tuning import (
        sweep_fusion_k_rrf,
        sweep_hybrid_k_rrf,
    )
    from src.marley.retrieval import BM25Retriever, HybridRetriever, VectorRetriever

    logger.info("=" * 60)
    logger.info("RRF K-PARAMETER TUNING")

    results = []

    # Hybrid sweep (per single KB)
    for kb, chunk_path in CHUNK_PATHS.items():
        eval_path = EVAL_PATHS[kb]
        logger.info("  Hybrid sweep: %s", kb)

        persist_base = output_dir / ".chromadb-rrf-tuning"

        def hybrid_factory(k_rrf, _kb=kb):
            return HybridRetriever(
                (
                    BM25Retriever(),
                    VectorRetriever(
                        persist_directory=str(persist_base / f"hybrid-{_kb}-{k_rrf}"),
                    ),
                ),
                k_rrf=k_rrf,
            )

        report = sweep_hybrid_k_rrf(
            retriever_factory=hybrid_factory,
            chunk_path=chunk_path,
            eval_path=eval_path,
        )
        report["knowledge_base"] = kb
        results.append(report)

    # Fusion sweep (all KBs)
    logger.info("  Fusion sweep: all KBs")
    chunk_paths = dict(CHUNK_PATHS)
    eval_paths = dict(EVAL_PATHS)
    report = sweep_fusion_k_rrf(
        retriever_factory=BM25Retriever,
        chunk_paths=chunk_paths,
        eval_paths=eval_paths,
    )
    results.append(report)

    _save_json(results, output_dir / "rrf-tuning.json")


def run_generation_step(output_dir: Path, ollama_url: str, ollama_model: str) -> None:
    """Run generation evaluation: single-KB + combined."""
    from evaluation.generation.evaluate import (
        run_and_report as run_gen_report,
    )
    from evaluation.utils import load_evaluation
    from src.marley.generator.ollama import OllamaGenerator
    from src.marley.retrieval import load_chunks

    logger.info("=" * 60)
    logger.info("GENERATION EVALUATION")

    generator = OllamaGenerator(model=ollama_model, base_url=ollama_url)
    reports = []

    for kb, chunk_path in CHUNK_PATHS.items():
        eval_path = EVAL_PATHS[kb]
        logger.info("  KB: %s", kb)
        chunks = load_chunks(chunk_path)
        questions = load_evaluation(eval_path)
        report = run_gen_report(generator, chunks, questions, knowledge_base=kb)
        reports.append(report)

    _save_json(reports, output_dir / "generation-evaluation.json")


def run_abstention_step(
    output_dir: Path,
    ollama_url: str,
    ollama_model: str,
) -> None:
    """Run abstention evaluation: Level 1 sweep + full evaluation."""
    from evaluation.abstention.evaluate import (
        run_abstention_evaluation,
        run_level1_sweep,
    )
    from evaluation.utils import load_evaluation
    from src.marley.generator.ollama import OllamaGenerator
    from src.marley.retrieval import BM25Retriever, load_chunks

    logger.info("=" * 60)
    logger.info("ABSTENTION EVALUATION")

    generator = OllamaGenerator(model=ollama_model, base_url=ollama_url)
    reports = []

    for kb, chunk_path in CHUNK_PATHS.items():
        eval_path = EVAL_PATHS[kb]
        logger.info("  KB: %s", kb)
        chunks = load_chunks(chunk_path)
        questions = load_evaluation(eval_path)
        retriever = BM25Retriever()
        retriever.index(chunks)

        # Level 1 sweep
        sweep = run_level1_sweep(
            retriever,
            chunks,
            questions,
            normalization_strategy="bm25",
        )

        # Full evaluation at best threshold
        best_threshold = max(sweep, key=lambda s: s["metrics"]["f1"])["threshold"]
        report = run_abstention_evaluation(
            retriever,
            generator,
            chunks,
            questions,
            threshold=best_threshold,
            normalization_strategy="bm25",
        )
        report["level1_sweep"] = sweep
        report["knowledge_base"] = kb
        reports.append(report)

    _save_json(reports, output_dir / "abstention-evaluation.json")


def run_e2e_step(
    output_dir: Path,
    ollama_url: str,
    ollama_model: str,
    config_filter: str | None = None,
) -> None:
    """Run end-to-end evaluation (delegates to existing run_all)."""
    from evaluation.end_to_end.run_all import run_all

    logger.info("=" * 60)
    logger.info("END-TO-END EVALUATION")

    run_all(
        output_dir=output_dir,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        config_filter=config_filter,
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Unified evaluation runner for the MARley pipeline.",
    )

    # Step selection
    parser.add_argument(
        "--check", action="store_true", help="Validate data requirements only"
    )
    parser.add_argument(
        "--retrieval", action="store_true", help="Run retrieval evaluation"
    )
    parser.add_argument(
        "--rrf-tuning", action="store_true", help="Sweep k_rrf for Hybrid and Fusion"
    )
    parser.add_argument(
        "--generation", action="store_true", help="Run generation evaluation"
    )
    parser.add_argument(
        "--abstention", action="store_true", help="Run abstention evaluation"
    )
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end evaluation")
    parser.add_argument("--all", action="store_true", help="Run all evaluation steps")

    # Common options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/testing",
        help="Output directory (default: data/testing)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--ollama-model", type=str, default="llama3.1:latest", help="Ollama model name"
    )
    parser.add_argument(
        "--config-filter",
        type=str,
        default=None,
        help="Only run E2E configs matching this substring",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Determine which steps to run
    if args.all:
        steps = list(ALL_STEPS)
    else:
        steps = []
        if args.retrieval:
            steps.append("retrieval")
        if args.rrf_tuning:
            steps.append("rrf-tuning")
        if args.generation:
            steps.append("generation")
        if args.abstention:
            steps.append("abstention")
        if args.e2e:
            steps.append("e2e")

    if args.check or not steps:
        # Validate and report
        check_steps = steps or ALL_STEPS
        errors = validate_data_requirements(
            check_steps,
            output_dir,
            args.ollama_url,
        )
        if errors:
            logger.error("Data validation failed:")
            for err in errors:
                logger.error("  ERROR: %s", err)
            sys.exit(1)
        else:
            logger.info(
                "All data requirements satisfied for: %s", ", ".join(check_steps)
            )
            if args.check:
                return

    # Validate before running
    errors = validate_data_requirements(steps, output_dir, args.ollama_url)
    if errors:
        logger.error("Data validation failed:")
        for err in errors:
            logger.error("  ERROR: %s", err)
        sys.exit(1)

    start = time.time()
    logger.info("Running evaluation steps: %s", ", ".join(steps))
    logger.info("Output directory: %s", output_dir)

    # Execute steps in order
    step_runners = {
        "retrieval": lambda: run_retrieval_step(output_dir),
        "rrf-tuning": lambda: run_rrf_tuning_step(output_dir),
        "generation": lambda: run_generation_step(
            output_dir,
            args.ollama_url,
            args.ollama_model,
        ),
        "abstention": lambda: run_abstention_step(
            output_dir,
            args.ollama_url,
            args.ollama_model,
        ),
        "e2e": lambda: run_e2e_step(
            output_dir,
            args.ollama_url,
            args.ollama_model,
            args.config_filter,
        ),
    }

    for step in steps:
        runner = step_runners[step]
        runner()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE in %.1fs", elapsed)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
