"""Reproduce the summary statistics cited in `analysis-ollama.md`.

Run from the project root::

    python data/evaluation-ollama-1.0/compute_stats.py

The script is dependency-light (standard library only) and reads the
committed JSON result files side-by-side so a reviewer can verify every
number that appears in the analysis notes without running the full
evaluation pipeline.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def _load(name: str) -> object:
    with (HERE / name).open(encoding="utf-8") as f:
        return json.load(f)


def retrieval_means() -> None:
    """Section 1 of analysis-ollama.md: mean F1@5 per retriever_type."""
    data = _load("retrieval-evaluation.json")
    by: dict[str, list[float]] = {}
    for entry in data:
        by.setdefault(entry["config"]["retriever_type"], []).append(
            entry["metrics"]["f1_at_k"],
        )
    print("Retrieval — mean F1@5 per retriever_type")
    for retr, values in sorted(by.items()):
        print(
            f"  {retr:<20s} mean={mean(values):.4f}  "
            f"range=[{min(values):.4f}, {max(values):.4f}]  n={len(values)}",
        )
    print()


def _per_result_scores(obj: dict, key: str) -> list[float]:
    out: list[float] = []
    for r in obj.get("results", []):
        val = r.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        out.append(f)
    return out


def e2e_scored_counts() -> tuple[int, list[tuple[str, int, float]]]:
    """E2E sample counts and per-config mean factual correctness.

    Results store per-sample judge scores under the `correctness`,
    `faithfulness`, and `answer_relevance` keys (the summary means live
    under the top-level `generation_metrics` block, but we recompute
    from raw samples so the reviewer can audit the aggregation).

    Returns (total_scored, [(config_name, n_scored, mean_fc), ...]).
    """
    rows: list[tuple[str, int, float]] = []
    total = 0
    for fpath in sorted(glob.glob(str(HERE / "e2e-results-*.json"))):
        with open(fpath, encoding="utf-8") as f:
            obj = json.load(f)
        fc_vals = _per_result_scores(obj, "correctness")
        name = Path(fpath).stem.replace("e2e-results-", "")
        n = len(fc_vals)
        total += n
        rows.append((name, n, mean(fc_vals) if fc_vals else float("nan")))
    return total, rows


def pearson(xs: list[float], ys: list[float]) -> float:
    """Plain Pearson product-moment correlation (no numpy dependency)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def abstention_f1_vs_fc_correlation() -> None:
    """Section 1 / §3 of analysis-ollama.md: Pearson r(Abst-F1, FC)."""
    xs: list[float] = []  # abstention F1
    ys: list[float] = []  # mean factual correctness
    skipped: list[str] = []
    for fpath in sorted(glob.glob(str(HERE / "e2e-results-*.json"))):
        with open(fpath, encoding="utf-8") as f:
            obj = json.load(f)
        am = obj.get("abstention_metrics") or {}
        abst_f1 = am.get("f1")
        fc_vals = _per_result_scores(obj, "correctness")
        if abst_f1 is None or not fc_vals:
            skipped.append(Path(fpath).stem)
            continue
        xs.append(float(abst_f1))
        ys.append(mean(fc_vals))
    r = pearson(xs, ys)
    print(
        "Pearson r(Abstention-F1, mean Factual Correctness) "
        f"across {len(xs)} E2E configs: {r:.3f}",
    )
    if skipped:
        print(
            f"  (skipped {len(skipped)} configs with no scored samples)",
        )
    print()


def rrf_best() -> None:
    """RRF tuning winners per sweep_type / KB (Section "RRF-Tuning")."""
    data = _load("rrf-tuning.json")
    if not isinstance(data, list):
        return
    print("RRF tuning — best k_rrf per (sweep_type, knowledge_base)")
    for entry in data:
        mode = entry.get("sweep_type") or "?"
        kb = entry.get("knowledge_base") or "all"
        best_k = entry.get("best_k_rrf")
        best_f1 = (entry.get("best_metrics") or {}).get("f1_at_k")
        f1_str = f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else str(best_f1)
        print(
            f"  sweep={mode:<7s} kb={kb:<10s} k_rrf={best_k} F1@5={f1_str}",
        )
    print()


def main() -> None:
    retrieval_means()
    total, rows = e2e_scored_counts()
    print(f"E2E — {len(rows)} configs, total scored samples: {total}")
    top5 = sorted(
        (r for r in rows if r[1] > 0),
        key=lambda t: (-t[2], -t[1]),
    )[:5]
    print("Top-5 configs by mean factual_correctness:")
    for name, n, fc in top5:
        print(f"  {name:<38s}  n={n:<3d}  mean_FC={fc:.3f}")
    print()
    abstention_f1_vs_fc_correlation()
    rrf_best()


if __name__ == "__main__":
    main()
