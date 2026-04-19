"""Tests for evaluation/validate.py prerequisites and CLI wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evaluation import validate as validate_mod
from evaluation.validate import _cli, validate_data_requirements


# ---------------------------------------------------------------------------
# validate_data_requirements
# ---------------------------------------------------------------------------


class TestValidateDataRequirements:
    def test_happy_path_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All chunk + eval files present, no Ollama needed → no errors."""
        monkeypatch.setattr(
            validate_mod, "CHUNK_PATHS", {"stpo": __file__},  # any existing file
        )
        monkeypatch.setattr(
            validate_mod, "EVAL_PATHS", {"stpo": __file__},
        )
        errors = validate_data_requirements(["retrieval"])
        assert errors == []

    def test_missing_chunk_file_produces_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_mod, "CHUNK_PATHS", {"stpo": "does/not/exist.json"},
        )
        monkeypatch.setattr(
            validate_mod, "EVAL_PATHS", {"stpo": __file__},
        )
        errors = validate_data_requirements(["retrieval"])
        assert any("Missing chunk file" in e for e in errors)

    def test_missing_eval_file_produces_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_mod, "CHUNK_PATHS", {"stpo": __file__},
        )
        monkeypatch.setattr(
            validate_mod, "EVAL_PATHS", {"stpo": "does/not/exist.json"},
        )
        errors = validate_data_requirements(["retrieval"])
        assert any("Missing evaluation file" in e for e in errors)

    def test_ollama_step_checks_ollama_availability(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            validate_mod, "CHUNK_PATHS", {"stpo": __file__},
        )
        monkeypatch.setattr(
            validate_mod, "EVAL_PATHS", {"stpo": __file__},
        )
        monkeypatch.setattr(
            validate_mod,
            "check_ollama",
            lambda url: {"available": False, "error": "connection refused"},
        )
        errors = validate_data_requirements(["e2e"])
        assert any("Ollama not available" in e for e in errors)

    def test_non_ollama_step_ignores_ollama(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            validate_mod, "CHUNK_PATHS", {"stpo": __file__},
        )
        monkeypatch.setattr(
            validate_mod, "EVAL_PATHS", {"stpo": __file__},
        )

        def _fail(_url: str) -> dict:
            raise AssertionError("check_ollama must not be called for 'retrieval'")

        monkeypatch.setattr(validate_mod, "check_ollama", _fail)
        errors = validate_data_requirements(["retrieval"])
        assert errors == []


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


class TestValidateCLI:
    def test_cli_exits_zero_when_all_ok(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            validate_mod, "validate_data_requirements", lambda *a, **kw: [],
        )
        assert _cli(["--steps", "retrieval"]) == 0
        captured = capsys.readouterr()
        assert "prerequisites satisfied" in captured.out

    def test_cli_exits_non_zero_and_prints_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            validate_mod,
            "validate_data_requirements",
            lambda *a, **kw: ["Missing: something"],
        )
        assert _cli(["--steps", "retrieval"]) == 1
        captured = capsys.readouterr()
        assert "Missing: something" in captured.err

    def test_cli_default_steps_cover_all(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_args: dict = {}

        def _fake(steps, output_dir, ollama_url):  # type: ignore[no-untyped-def]
            captured_args["steps"] = steps
            return []

        monkeypatch.setattr(
            validate_mod, "validate_data_requirements", _fake,
        )
        assert _cli([]) == 0
        assert set(captured_args["steps"]) == {
            "retrieval", "rrf-tuning", "generation", "abstention", "e2e",
        }
