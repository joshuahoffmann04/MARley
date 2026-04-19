"""Unified evaluation CLI for the MARley pipeline.

Single entry point for all evaluation steps:

    python -m evaluation --check                     # Validate data requirements
    python -m evaluation --retrieval                 # Run retrieval evaluation
    python -m evaluation --rrf-tuning                # Sweep k_rrf for Hybrid and Fusion
    python -m evaluation --generation                # Run generation evaluation
    python -m evaluation --generation --judge openai # Use OpenAI gpt-4o-mini as judge
    python -m evaluation --abstention                # Run abstention evaluation
    python -m evaluation --e2e                       # Run end-to-end evaluation
    python -m evaluation --e2e --judge openai        # Same, with OpenAI judge for E2E scoring
    python -m evaluation --all --judge openai        # Full run, OpenAI judge for generation + E2E

The pipeline runs on GPU (CUDA baseline); the runner fails fast if CUDA
is unavailable. The ``--judge`` flag affects every step that scores
free-form answers with RAGAS — currently ``--generation`` and
``--e2e`` — and, by extension, the generation and E2E steps of
``--all``. Abstention, retrieval, and RRF tuning measure deterministic
booleans and set operations; they ignore the flag.

Execution order: retrieval -> rrf-tuning -> generation -> abstention -> e2e.
"""

from __future__ import annotations

# Load .env before anything else so that OPENAI_API_KEY is visible to
# every downstream import (notably the judge factory).
from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

from evaluation.judge import Judge, make_judge
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


def _parse_distractor_levels(raw: str | None) -> list[int] | None:
    """Parse a comma-separated distractor-level string.

    Returns ``None`` when unset so downstream code applies its default
    (``0..10``). Invalid input aborts with a clear argparse-style error.
    """
    if raw is None or raw.strip() == "":
        return None
    try:
        levels = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(
            f"--distractor-levels must be comma-separated integers, got {raw!r}"
        ) from exc
    if any(n < 0 for n in levels):
        raise SystemExit(
            f"--distractor-levels must be non-negative integers, got {raw!r}"
        )
    return levels


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _require_cuda() -> None:
    """Abort unless CUDA is available.

    The pipeline treats GPU as the baseline, not an optimisation. A missing
    GPU means the evaluation cannot run — bail out with an actionable hint.
    """
    if not torch.cuda.is_available():
        logger.error(
            "GPU required. Install CUDA PyTorch: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)
    logger.info(
        "GPU detected: %s (torch %s)",
        torch.cuda.get_device_name(0),
        torch.__version__,
    )


def _require_openai_key(judge_backend: str) -> None:
    """Abort if the OpenAI backend is requested but no API key is set."""
    if judge_backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY not set. Add it to .env or use --judge ollama."
        )
        sys.exit(1)


def _warn_ollama_parallel() -> None:
    """Warn if ``OLLAMA_NUM_PARALLEL`` is not set to 2.

    Two slots are the throughput sweet spot: RAGAS batches fill both slots
    without queueing, while more slots contend on the same GPU and stall.
    """
    value = os.environ.get("OLLAMA_NUM_PARALLEL")
    if value != "2":
        logger.warning(
            "OLLAMA_NUM_PARALLEL=%s detected. For best performance, "
            "set to 2 and restart Ollama.",
            value or "<unset>",
        )


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


def run_generation_step(
    output_dir: Path,
    ollama_url: str,
    ollama_model: str,
    judge: Judge,
    *,
    subset: int | None = None,
    distractor_levels: list[int] | None = None,
    kb_filter: str | None = None,
) -> None:
    """Run generation evaluation: single-KB + combined, scored via the judge.

    Args:
        output_dir: Where JSON reports are written.
        ollama_url: URL of the Ollama server powering the generator.
        ollama_model: Ollama model used as the generator (never the judge).
        judge: Pre-constructed judge (Ollama or OpenAI) for RAGAS scoring.
        subset: Optional cap on number of questions (per KB) — useful for
            quick subset verification runs. ``None`` means full eval set.
        distractor_levels: Optional override for distractor counts, e.g.
            ``[0, 5, 10]``. ``None`` means the library default ``0..10``.
        kb_filter: Optional single-KB name. When set, only this KB is run
            and the combined-KB evaluation is skipped.
    """
    from evaluation.generation.combined import run_and_report_combined
    from evaluation.generation.evaluate import (
        run_and_report as run_gen_report,
    )
    from src.marley.generator.ollama import OllamaGenerator
    from src.marley.retrieval import load_chunks

    logger.info("=" * 60)
    logger.info("GENERATION EVALUATION (judge batch_size=%d)", judge.batch_size)

    generator = OllamaGenerator(model=ollama_model, base_url=ollama_url)
    reports = []

    kbs = {kb_filter: CHUNK_PATHS[kb_filter]} if kb_filter else CHUNK_PATHS

    # Single-KB evaluation
    for kb, chunk_path in kbs.items():
        eval_path = EVAL_PATHS[kb]
        logger.info("  Single-KB: %s", kb)
        chunks = load_chunks(chunk_path)
        report = run_gen_report(
            generator, chunks, eval_path, judge,
            distractor_levels=distractor_levels,
            knowledge_base=kb,
            subset=subset,
        )
        reports.append(report)

    _save_json(reports, output_dir / "generation-evaluation.json")

    # Combined-KB evaluation (skipped on subset or single-KB filter runs —
    # those are verification shortcuts, not production reports).
    if kb_filter is None and subset is None:
        logger.info("  Combined-KB: stpo + faq-stpo + faq-ao")
        combined_chunk_paths = dict(CHUNK_PATHS)
        combined_eval_paths = dict(EVAL_PATHS)
        combined_report = run_and_report_combined(
            generator,
            combined_chunk_paths,
            combined_eval_paths,
            judge,
            distractor_levels=distractor_levels,
        )
        _save_json(combined_report, output_dir / "generation-evaluation-combined.json")


def run_abstention_step(
    output_dir: Path,
    ollama_url: str,
    ollama_model: str,
) -> None:
    """Run abstention evaluation: Level 1 sweep + full evaluation.

    Abstention uses deterministic metrics (regex + score thresholds), no
    LLM judge, so the ``--judge`` flag does not apply here.
    """
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

        # Full evaluation at best threshold (F0.5-optimal, precision-weighted)
        best_threshold = max(sweep, key=lambda s: s["metrics"]["f0_5"])["threshold"]
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
    judge: Judge,
    config_filter: str | None = None,
) -> None:
    """Run end-to-end evaluation.

    For each of the 33 configurations, runs the full pipeline and scores
    every non-abstained answerable answer with RAGAS via ``judge``.
    Abstention metrics (deterministic) and generation metrics (RAGAS)
    appear side by side in the per-config report.
    """
    from evaluation.end_to_end.run_all import run_all

    logger.info("=" * 60)
    logger.info("END-TO-END EVALUATION (judge batch_size=%d)", judge.batch_size)

    run_all(
        output_dir=output_dir,
        judge=judge,
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
        default="data/evaluation",
        help="Output directory (default: data/evaluation)",
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
        "--ollama-judge-model",
        type=str,
        default=None,
        help=(
            "Optional Ollama model used as the RAGAS judge when "
            "--judge=ollama. Defaults to --ollama-model. A larger model "
            "(e.g. qwen2.5:14b) typically produces fewer NaN scores and "
            "better-calibrated Factual Correctness verdicts than the 8B "
            "generator. Must be pulled on the Ollama server first."
        ),
    )
    parser.add_argument(
        "--config-filter",
        type=str,
        default=None,
        help="Only run E2E configs matching this substring",
    )

    # Judge + generation-specific options
    parser.add_argument(
        "--judge",
        choices=["ollama", "openai"],
        default="ollama",
        help=(
            "RAGAS judge backend for answer scoring. "
            "Default: ollama (local). Affects --generation, --e2e, and the "
            "corresponding steps of --all. --judge openai uses gpt-4o-mini "
            "and requires OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help=(
            "Limit the generation eval to the first N questions per KB. "
            "Useful for quick subset-verification runs. Generation-only."
        ),
    )
    parser.add_argument(
        "--distractor-levels",
        type=str,
        default=None,
        help=(
            "Comma-separated distractor counts for the generation step, "
            "e.g. '0,5,10'. Default is the full sweep 0..10. "
            "Generation-only."
        ),
    )
    parser.add_argument(
        "--kb-filter",
        type=str,
        default=None,
        choices=["stpo", "faq-stpo", "faq-ao"],
        help=(
            "Restrict the generation step to a single knowledge base and "
            "skip the combined-KB run. Generation-only."
        ),
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    distractor_levels = _parse_distractor_levels(args.distractor_levels)

    # GPU is the baseline — abort early if it's missing.
    _require_cuda()

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

    # Build the judge lazily — only when at least one judge-consuming
    # step (generation, e2e) is scheduled, and only after the required
    # secrets are confirmed present. Both steps share the same Judge
    # instance so the HuggingFaceEmbeddings model is loaded once.
    judge: Judge | None = None
    if "generation" in steps or "e2e" in steps:
        _require_openai_key(args.judge)
        _warn_ollama_parallel()
        logger.info("Judge backend: %s", args.judge)
        judge = make_judge(
            args.judge,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            ollama_judge_model=args.ollama_judge_model,
        )
        if args.ollama_judge_model and args.judge == "ollama":
            logger.info(
                "Judge model override: %s (generator remains %s)",
                args.ollama_judge_model, args.ollama_model,
            )

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
            judge,
            subset=args.subset,
            distractor_levels=distractor_levels,
            kb_filter=args.kb_filter,
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
            judge,
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
