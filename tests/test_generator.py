import pytest
from unittest.mock import MagicMock, patch
from generator.prompts import PromptBuilder
from generator.service import GeneratorService
from generator.config import GeneratorSettings, get_generator_config
from generator.models import GenerateRequest, LLMStructuredOutput, RetrievalSearchResponse, RetrievalHit

def test_prompt_builder_defaults():
    builder = PromptBuilder()
    sys_prompt = builder.build_system_prompt()
    assert "MARley" in sys_prompt
    
    user_prompt = builder.build_user_prompt("Was ist X?", "Kontext Y")
    assert "Was ist X?" in user_prompt
    assert "Kontext Y" in user_prompt

def test_generator_service_init():
    settings = GeneratorSettings(
        document_id="doc1",
        ollama_base_url="http://localhost:11434"
    )
    config = get_generator_config(settings)
    service = GeneratorService(settings=settings, runtime_config=config)
    assert service is not None

def test_generator_service_generate_flow():
    settings = GeneratorSettings(
        document_id="doc1",
        ollama_base_url="http://localhost:11434",
        retrieval_base_url="http://localhost:8000",
        min_hits_default=1,
        min_dual_backend_hits_default=0   
    )
    config = get_generator_config(settings)
    service = GeneratorService(settings=settings, runtime_config=config)
    
    # Mock LLMClient
    service.llm_client.chat_structured = MagicMock(return_value=(
        LLMStructuredOutput(
            answer="Test answer",
            should_abstain=False,
            confidence="high",
            reasoning="Test reasoning"
        ),
        '{"answer": "Test answer"}',
        None
    ))
    
    # Mock Retrieval Call
    mock_retrieval_resp = RetrievalSearchResponse(
        document_id="doc1",
        query="test query",
        top_k=2,
        hits=[
            RetrievalHit(
                rank=1, rrf_score=1.0, source_type="pdf", chunk_id="c1", chunk_type="text",
                text="Content 1", metadata={}, input_file="f1"
            )
        ],
        quality_flags=[]
    )
    service._call_retrieval_search = MagicMock(return_value=mock_retrieval_resp)

    req = GenerateRequest(query="test query", document_id="doc1")
    resp = service.generate(req)
    
    assert resp.answer == "Test answer"
    assert not resp.abstained
    assert len(resp.used_chunks) == 1
    assert resp.used_chunks[0].chunk_id == "c1"
    
    # Verify mock calls
    service._call_retrieval_search.assert_called_once()
    service.llm_client.chat_structured.assert_called_once()
