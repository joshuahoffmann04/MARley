"""Data model for the manual evaluation framework.

Defines the core types (EvaluationItem, ManualJudgement, Judgement enum)
and I/O functions for loading and saving evaluation data. Items and
judgements are stored in separate JSON files — items are immutable,
judgements are append-only and incrementally saved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Judgement schema
# ---------------------------------------------------------------------------


class Judgement(str, Enum):
    """Possible judgement values for manual evaluation.

    Answer judgements (for answerable questions):
        CORRECT: Answer is semantically correct and complete.
        PARTIALLY_CORRECT: Answer has some correct info but is incomplete
            or has minor issues.
        INCORRECT: Answer is factually wrong, hallucinates, or misses
            the core information.

    Abstention judgements:
        CORRECT_ABSTENTION: System correctly refused to answer an
            unanswerable question.
        INCORRECT_ABSTENTION: System refused to answer but should have
            been able to (false negative).
        MISSING_ABSTENTION: System provided an answer but should have
            abstained (false positive).
    """

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    CORRECT_ABSTENTION = "correct_abstention"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    MISSING_ABSTENTION = "missing_abstention"


#: Judgement values for answerable questions.
ANSWER_JUDGEMENTS = frozenset({
    Judgement.CORRECT,
    Judgement.PARTIALLY_CORRECT,
    Judgement.INCORRECT,
})

#: Judgement values for abstention assessment.
ABSTENTION_JUDGEMENTS = frozenset({
    Judgement.CORRECT_ABSTENTION,
    Judgement.INCORRECT_ABSTENTION,
    Judgement.MISSING_ABSTENTION,
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationItem:
    """A single item to be judged by the human evaluator.

    Generic across evaluation types. The ``metadata`` dict holds
    evaluation-type-specific fields (e.g. ``num_distractors`` for
    generation evaluation).
    """

    id: str
    question: str
    generated_answer: str
    reference_answer: str
    category: str
    expected_abstention: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class ManualJudgement:
    """A single human judgement for an evaluation item.

    If an item is judged multiple times, the latest timestamp wins.
    """

    item_id: str
    judgement: Judgement
    notes: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if isinstance(self.judgement, str):
            self.judgement = Judgement(self.judgement)


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def save_items(
    items: list[EvaluationItem],
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save evaluation items to a JSON file.

    Args:
        items: List of evaluation items to save.
        path: Output file path.
        metadata: Optional metadata dict to include in the file header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata or {},
        "items": [asdict(item) for item in items],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_items(path: str | Path) -> list[EvaluationItem]:
    """Load evaluation items from a JSON file.

    Returns:
        List of EvaluationItem instances.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    return [
        EvaluationItem(**item_dict)
        for item_dict in data["items"]
    ]


def save_judgement(judgement: ManualJudgement, path: str | Path) -> None:
    """Append a single judgement to the judgements file.

    Creates the file with an empty judgements list if it does not exist.
    The judgement is appended to the ``judgements`` array — if the same
    ``item_id`` appears multiple times, ``load_judgements`` will
    deduplicate by keeping the latest entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "metadata": {
                "started": datetime.now(timezone.utc).isoformat(),
                "last_updated": "",
            },
            "judgements": [],
        }

    data["judgements"].append(asdict(judgement))
    data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_judgements(path: str | Path) -> list[ManualJudgement]:
    """Load judgements from a JSON file, deduplicating by item_id.

    If an item_id appears multiple times, only the latest entry
    (last occurrence in file order) is kept. This allows corrections
    by simply appending a new judgement for the same item.

    Returns:
        List of ManualJudgement instances (one per unique item_id).
    """
    path = Path(path)
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))

    # Deduplicate: later entries override earlier ones
    seen: dict[str, ManualJudgement] = {}
    for j_dict in data["judgements"]:
        j = ManualJudgement(**j_dict)
        seen[j.item_id] = j

    return list(seen.values())
