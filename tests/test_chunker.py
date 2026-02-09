import pytest
from chunker.pdf_chunker.service import _split_oversized_sentence, _split_sentences
from chunker.pdf_chunker.models import PDFChunk, ChunkMetadata
import tiktoken

def test_split_sentences():
    text = "Dies ist ein Satz. Dies ist noch einer!"
    sentences = _split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "Dies ist ein Satz."

def test_chunk_consistency():
    # Verify PDFChunk follows BaseChunk structure
    meta = ChunkMetadata(
        document_id="doc1",
        source_file="test.pdf",
        chunk_index=0,
        section_id="sec1"
    )
    chunk = PDFChunk(
        chunk_id="ch_1",
        chunk_type="text",
        text="Hello World",
        token_count=2,
        metadata=meta
    )
    assert chunk.chunk_id == "ch_1"
    assert chunk.metadata.document_id == "doc1"
    # Check if we can dump to json (pydantic check)
    json_out = chunk.model_dump_json()
    assert "doc1" in json_out

def test_faq_chunk_consistency():
    from chunker.faq_chunker.models import FAQChunk, FAQChunkMetadata
    meta = FAQChunkMetadata(
        document_id="faq1",
        source_file="faq.json",
        chunk_index=0,
        faq_source="sb",
        faq_id="1"
    )
    chunk = FAQChunk(
        chunk_id="faq_sb_1",
        text="Q: A?",
        token_count=4,
        metadata=meta
    )
    assert chunk.chunk_type == "faq"
    assert chunk.metadata.faq_id == "1"
