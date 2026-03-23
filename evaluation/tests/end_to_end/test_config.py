"""Tests for end-to-end evaluation configuration."""

from evaluation.end_to_end.config import (
    E2EConfig,
    KNOWLEDGE_BASES,
    NORMALIZATION_MAP,
    generate_all_configs,
)


class TestE2EConfig:
    """Tests for the E2EConfig dataclass."""

    def test_config_is_frozen_and_hashable(self):
        config = E2EConfig(
            name="test",
            retriever_type="bm25",
            knowledge_bases=("stpo",),
            strategy="single",
            normalization_strategy="bm25",
        )
        # hashable
        assert hash(config) is not None
        # frozen
        try:
            config.name = "changed"
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_fields_stored_correctly(self):
        config = E2EConfig(
            name="single-stpo-bm25",
            retriever_type="bm25",
            knowledge_bases=("stpo",),
            strategy="single",
            normalization_strategy="bm25",
            k=10,
            k_rrf=30,
        )
        assert config.name == "single-stpo-bm25"
        assert config.retriever_type == "bm25"
        assert config.knowledge_bases == ("stpo",)
        assert config.strategy == "single"
        assert config.normalization_strategy == "bm25"
        assert config.k == 10
        assert config.k_rrf == 30

    def test_equality_comparison(self):
        a = E2EConfig(
            name="test", retriever_type="bm25",
            knowledge_bases=("stpo",), strategy="single",
            normalization_strategy="bm25",
        )
        b = E2EConfig(
            name="test", retriever_type="bm25",
            knowledge_bases=("stpo",), strategy="single",
            normalization_strategy="bm25",
        )
        assert a == b


class TestGenerateAllConfigs:
    """Tests for generate_all_configs()."""

    def test_total_count_is_33(self):
        configs = generate_all_configs()
        assert len(configs) == 33

    def test_9_single_configs(self):
        configs = generate_all_configs()
        single = [c for c in configs if c.strategy == "single"]
        assert len(single) == 9

    def test_12_merged_pool_configs(self):
        configs = generate_all_configs()
        merged = [c for c in configs if c.strategy == "merged_pool"]
        assert len(merged) == 12

    def test_12_fusion_configs(self):
        configs = generate_all_configs()
        fusion = [c for c in configs if c.strategy == "fusion"]
        assert len(fusion) == 12

    def test_all_names_unique(self):
        configs = generate_all_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names))

    def test_fusion_configs_always_use_rrf_normalization(self):
        configs = generate_all_configs()
        fusion = [c for c in configs if c.strategy == "fusion"]
        for c in fusion:
            assert c.normalization_strategy == "rrf"

    def test_single_and_merged_normalization_matches_retriever(self):
        configs = generate_all_configs()
        non_fusion = [c for c in configs if c.strategy != "fusion"]
        for c in non_fusion:
            assert c.normalization_strategy == NORMALIZATION_MAP[c.retriever_type]
