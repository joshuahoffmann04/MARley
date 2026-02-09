import json
import pytest
from pathlib import Path
from retrieval.base import SearchRequest
from retrieval.sparse_retrieval.service import SparseBM25Retriever
from retrieval.sparse_retrieval.config import SparseRetrievalSettings, get_sparse_retrieval_config

@pytest.fixture
def test_env(tmp_path):
    doc_id = "test_doc"
    doc_dir = tmp_path / doc_id
    chunks_dir = doc_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    
    # Create a dummy chunk file
    chunk_file = chunks_dir / f"{doc_id}-pdf-chunker-output.json"
    payload = {
        "chunks": [
            {
                "chunk_id": "c1",
                "chunk_type": "text",
                "text": "This is a test chunk for retrieval.",
                "token_count": 6,
                "metadata": {"page": 1}
            },
            {
                "chunk_id": "c2",
                "chunk_type": "text",
                "text": "Just another chunk without the keyword.",
                "token_count": 5,
                "metadata": {"page": 2}
            },
            {
                "chunk_id": "c3",
                "chunk_type": "text",
                "text": "Third chunk effectively making the term rare.",
                "token_count": 5,
                "metadata": {"page": 3}
            }
        ]
    }
    chunk_file.write_text(json.dumps(payload), encoding="utf-8")
    
    return tmp_path, doc_id

def test_sparse_retrieval_success(test_env):
    data_root, doc_id = test_env
    settings = SparseRetrievalSettings(
        document_id=doc_id,
        data_root=data_root,
        app_name="TestApp"
    )
    # Force input_dir to be the chunks dir we created, or let it resolve?
    # SparseBM25Retriever resolves input_dir based on data_root/doc_id/chunks
    # So it should work.
    
    config = get_sparse_retrieval_config(settings)
    retriever = SparseBM25Retriever(settings=settings, runtime_config=config)
    
    req = SearchRequest(
        query="retrieval",
        document_id=doc_id,
        rebuild_if_stale=True
    )
    print(f"Chunks dir: {chunks_dir}")
    print(f"Files in chunks dir: {list(chunks_dir.iterdir())}")
    
    try:
        resp = retriever.search(req)
        assert len(resp.hits) == 1
        assert resp.hits[0].chunk_id == "c1"
        assert "retrieval" in resp.hits[0].content
    except Exception as e:
        print(f"\nSearch failed with error: {e}")
        # Re-raise to fail the test but we saw output
        raise
