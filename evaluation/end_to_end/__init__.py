"""End-to-end evaluation for the MARley pipeline."""

from evaluation.end_to_end.config import E2EConfig, generate_all_configs
from evaluation.end_to_end.evaluate import (
    E2EResult,
    run_and_report,
    run_e2e_config,
    save_report,
    sweep_threshold,
)
from evaluation.end_to_end.metrics import (
    E2EConfigMetrics,
    build_comparison_table,
    compute_e2e_config_metrics,
)
from evaluation.end_to_end.run_all import run_all

__all__ = [
    "E2EConfig",
    "E2EConfigMetrics",
    "E2EResult",
    "build_comparison_table",
    "compute_e2e_config_metrics",
    "generate_all_configs",
    "run_all",
    "run_and_report",
    "run_e2e_config",
    "save_report",
    "sweep_threshold",
]
