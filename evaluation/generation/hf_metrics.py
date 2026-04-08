"""HuggingFace evaluate metrics for generation quality assessment.

Wraps the HuggingFace ``evaluate`` library to compute ROUGE and BERTScore
metrics for a list of (prediction, reference) pairs.

Both metrics are loaded lazily and cached within the module to avoid
repeated model loading across multiple evaluation calls.
"""

from __future__ import annotations

import evaluate as hf_evaluate

_rouge = None
_bertscore = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = hf_evaluate.load("rouge")
    return _rouge


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        _bertscore = hf_evaluate.load("bertscore")
    return _bertscore


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> list[dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 for each prediction/reference pair.

    Args:
        predictions: List of generated answers.
        references: List of reference answers (same length as predictions).

    Returns:
        List of dicts, each with keys 'rouge1', 'rouge2', 'rougeL'.
        Values are per-sentence F1 scores in [0.0, 1.0].
    """
    if not predictions:
        return []

    rouge = _get_rouge()
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False,
    )
    # result is dict of lists: {"rouge1": [...], "rouge2": [...], "rougeL": [...]}
    n = len(predictions)
    return [
        {
            "rouge1": float(result["rouge1"][i]),
            "rouge2": float(result["rouge2"][i]),
            "rougeL": float(result["rougeL"][i]),
        }
        for i in range(n)
    ]


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> list[float]:
    """Compute BERTScore F1 for each prediction/reference pair.

    Args:
        predictions: List of generated answers.
        references: List of reference answers (same length as predictions).
        lang: Language code for the BERTScore model (default: 'en').

    Returns:
        List of BERTScore F1 values in [0.0, 1.0], one per pair.
    """
    if not predictions:
        return []

    bertscore = _get_bertscore()
    result = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=lang,
    )
    return [float(v) for v in result["f1"]]
