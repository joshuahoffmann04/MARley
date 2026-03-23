"""End-to-end evaluation configuration.

Defines the E2EConfig dataclass and generates the full matrix of
33 pipeline configurations to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.marley.models.constants import (
    DEFAULT_K,
    DEFAULT_K_RRF,
    DEFAULT_K_RRF_FUSION,
    DEFAULT_K_RRF_HYBRID,
    NORMALIZATION_MAP,
    RETRIEVER_TYPES,
)

# Knowledge base identifiers (must match chunk/eval file naming)
KNOWLEDGE_BASES = ["stpo", "faq-stpo", "faq-ao"]

# All KB combinations for combined evaluation
KB_COMBINATIONS: list[list[str]] = [
    ["stpo", "faq-stpo"],
    ["stpo", "faq-ao"],
    ["faq-stpo", "faq-ao"],
    ["stpo", "faq-stpo", "faq-ao"],
]


@dataclass(frozen=True)
class E2EConfig:
    """A single end-to-end evaluation configuration."""

    name: str
    retriever_type: str
    knowledge_bases: tuple[str, ...]  # Tuple for hashability (frozen)
    strategy: str                     # "single", "merged_pool", "fusion"
    normalization_strategy: str
    k: int = DEFAULT_K
    k_rrf: int = DEFAULT_K_RRF  # Used for fusion strategy
    k_rrf_hybrid: int = DEFAULT_K_RRF_HYBRID  # Used for hybrid within fusion


def _kb_label(knowledge_bases: list[str]) -> str:
    """Short label for a KB combination."""
    if len(knowledge_bases) == len(KNOWLEDGE_BASES):
        return "all"
    return "+".join(knowledge_bases)


def generate_all_configs(
    k: int = DEFAULT_K,
    k_rrf: int = DEFAULT_K_RRF,
    k_rrf_hybrid: int = DEFAULT_K_RRF_HYBRID,
) -> list[E2EConfig]:
    """Generate all 33 end-to-end evaluation configurations.

    Returns:
        List of E2EConfig in deterministic order:
        - 9 single-KB (3 KBs x 3 retrievers)
        - 12 merged pool (4 combos x 3 retrievers)
        - 12 fusion (4 combos x 3 retrievers)
    """
    configs: list[E2EConfig] = []

    # Single-KB configurations
    for kb in KNOWLEDGE_BASES:
        for rt in RETRIEVER_TYPES:
            configs.append(E2EConfig(
                name=f"single-{kb}-{rt}",
                retriever_type=rt,
                knowledge_bases=(kb,),
                strategy="single",
                normalization_strategy=NORMALIZATION_MAP[rt],
                k=k,
                k_rrf=k_rrf,
                k_rrf_hybrid=k_rrf_hybrid,
            ))

    # Combined-KB: Merged Pool
    for combo in KB_COMBINATIONS:
        label = _kb_label(combo)
        for rt in RETRIEVER_TYPES:
            configs.append(E2EConfig(
                name=f"merged-{label}-{rt}",
                retriever_type=rt,
                knowledge_bases=tuple(combo),
                strategy="merged_pool",
                normalization_strategy=NORMALIZATION_MAP[rt],
                k=k,
                k_rrf=k_rrf,
                k_rrf_hybrid=k_rrf_hybrid,
            ))

    # Combined-KB: Fusion
    for combo in KB_COMBINATIONS:
        label = _kb_label(combo)
        for rt in RETRIEVER_TYPES:
            configs.append(E2EConfig(
                name=f"fusion-{label}-{rt}",
                retriever_type=rt,
                knowledge_bases=tuple(combo),
                strategy="fusion",
                normalization_strategy="rrf",  # Always RRF for fusion
                k=k,
                k_rrf=k_rrf,
                k_rrf_hybrid=k_rrf_hybrid,
            ))

    return configs
