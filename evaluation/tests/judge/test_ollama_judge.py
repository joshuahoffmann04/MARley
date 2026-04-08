from __future__ import annotations
import json
from unittest.mock import MagicMock, patch
import pytest
from evaluation.judge.base import Judge, JudgementResult
from evaluation.judge.ollama_judge import OllamaJudge, _parse_scores

CONTEXT = [
    {"chunk_id": "c1", "text": "The standard study period is 4 semesters."},
]

def _mock_response(content, model="llama3.1:8b"):
    resp = MagicMock()
    resp.message.content = content
    resp.model = model
    return resp

def _json_scores(f=0.9, r=0.85, c=0.8):
    return json.dumps({"faithfulness": f, "answer_relevance": r, "correctness": c})

class TestParseScores:
    def test_valid_json(self):
        raw = json.dumps({"faithfulness": 0.9, "answer_relevance": 0.8, "correctness": 0.7})
        scores = _parse_scores(raw)
        assert scores["faithfulness"] == pytest.approx(0.9)
        assert scores["answer_relevance"] == pytest.approx(0.8)
        assert scores["correctness"] == pytest.approx(0.7)
    def test_clamps_above_one(self):
        raw = json.dumps({"faithfulness": 1.5, "answer_relevance": 0.8, "correctness": 0.7})
        assert _parse_scores(raw)["faithfulness"] == pytest.approx(1.0)
    def test_clamps_below_zero(self):
        raw = json.dumps({"faithfulness": -0.1, "answer_relevance": 0.8, "correctness": 0.7})
        assert _parse_scores(raw)["faithfulness"] == pytest.approx(0.0)
    def test_missing_key_defaults_to_zero(self):
        raw = json.dumps({"faithfulness": 0.9})
        scores = _parse_scores(raw)
        assert scores["answer_relevance"] == pytest.approx(0.0)
        assert scores["correctness"] == pytest.approx(0.0)
    def test_json_embedded_in_prose(self):
        raw = str({"faithfulness": 0.9, "answer_relevance": 0.8, "correctness": 0.7}).replace("'", '"')
        scores = _parse_scores(raw)
        assert scores["faithfulness"] == pytest.approx(0.9)
    def test_invalid_json_returns_zeros(self):
        scores = _parse_scores("not valid json")
        assert scores == {"faithfulness": 0.0, "answer_relevance": 0.0, "correctness": 0.0}
    def test_non_numeric_value_defaults_to_zero(self):
        raw = json.dumps({"faithfulness": "high", "answer_relevance": 0.8, "correctness": 0.7})
        assert _parse_scores(raw)["faithfulness"] == pytest.approx(0.0)

class TestOllamaJudgeUnit:
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_implements_judge_interface(self, _mock):
        assert isinstance(OllamaJudge(), Judge)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_model_property(self, _mock):
        assert OllamaJudge(model="mistral:7b").model == "mistral:7b"
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_model_is_property(self, _mock):
        assert isinstance(type(OllamaJudge()).model, property)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_returns_judgement_result(self, mock_cls):
        mock_cls.return_value.chat.return_value = _mock_response(_json_scores())
        result = OllamaJudge().judge("q1", "How long?", CONTEXT, "4 semesters.", "4 semesters.")
        assert isinstance(result, JudgementResult)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_scores_parsed(self, mock_cls):
        mock_cls.return_value.chat.return_value = _mock_response(_json_scores(f=0.95, r=0.90, c=0.85))
        result = OllamaJudge().judge("q1", "q?", CONTEXT, "a.", "a.")
        assert result.faithfulness == pytest.approx(0.95)
        assert result.answer_relevance == pytest.approx(0.90)
        assert result.correctness == pytest.approx(0.85)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_propagates_question_id(self, mock_cls):
        mock_cls.return_value.chat.return_value = _mock_response(_json_scores())
        result = OllamaJudge().judge("eval-999", "q?", CONTEXT, "a.", "a.")
        assert result.question_id == "eval-999"
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_records_model(self, mock_cls):
        mock_cls.return_value.chat.return_value = _mock_response(_json_scores(), model="llama3.1:8b")
        result = OllamaJudge().judge("q1", "q?", CONTEXT, "a.", "a.")
        assert result.model == "llama3.1:8b"
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_abstained_returns_sentinel(self, _mock):
        result = OllamaJudge().judge("q1", "q?", CONTEXT, "ABSTENTION: no info", "a.")
        assert result.faithfulness == pytest.approx(1.0)
        assert result.answer_relevance == pytest.approx(0.0)
        assert result.correctness == pytest.approx(0.0)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_empty_answer_sentinel(self, _mock):
        result = OllamaJudge().judge("q1", "q?", CONTEXT, "", "a.")
        assert result.faithfulness == pytest.approx(1.0)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_abstention_case_insensitive(self, _mock):
        result = OllamaJudge().judge("q1", "q?", CONTEXT, "abstention: reason", "a.")
        assert result.faithfulness == pytest.approx(1.0)
    @patch("evaluation.judge.ollama_judge.ollama_lib.Client")
    def test_chat_uses_json_format(self, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.chat.return_value = _mock_response(_json_scores())
        OllamaJudge().judge("q1", "q?", CONTEXT, "a.", "a.")
        assert mock_client.chat.call_args.kwargs.get("format") == "json"

def _ollama_available():
    try:
        import ollama; ollama.list(); return True
    except Exception:
        return False

import pytest as _pytest
@_pytest.mark.skipif(not _ollama_available(), reason="Ollama server not running")
@_pytest.mark.integration
class TestOllamaJudgeIntegration:
    def test_judge_scores_in_range(self):
        judge = OllamaJudge()
        result = judge.judge("eval-001", "How long is the standard study period?", CONTEXT, "The standard study period is 4 semesters.", "4 semesters.")
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0
        assert 0.0 <= result.correctness <= 1.0