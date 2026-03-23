"""Manual evaluation framework for the MARley pipeline.

Provides a web UI for human correctness assessment of generated
answers, replacing the LLM-as-Judge approach with more reliable
and defensible manual evaluation.

Public API::

    from evaluation.manual.models import (
        Judgement, EvaluationItem, ManualJudgement,
        load_items, save_items, load_judgements, save_judgement,
    )
    from evaluation.manual.prepare import prepare_generation_items
    from evaluation.manual.metrics import compute_manual_metrics
"""
