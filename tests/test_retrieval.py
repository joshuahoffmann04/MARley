import pytest
from retrieval.base import SearchRequest, SearchResponse
from retrieval.sparse_retrieval.service import SparseBM25Retriever
from retrieval.sparse_retrieval.config import SparseRetrievalSettings, get_sparse_retrieval_config

# Mock or use temporary directory for tests
@pytest.fixture
def sparse_retriever(tmp_path):
    settings = SparseRetrievalSettings(
        document_id="test_doc",
        data_root=tmp_path
    )
    config = get_sparse_retrieval_config(settings)
    return SparseBM25Retriever(settings=settings, runtime_config=config)

def test_sparse_search_request_handling(sparse_retriever):
    # This test primarily checks if the refactored method signature works
    # It will fail to find chunks in tmp_path, but should raise proper exception or return empty
    
    # We expect ValueError("No valid chunks...") if no chunks found during implicit index build
    request = SearchRequest(
        query="test",
        document_id="test_doc"
    )
    
    # Needs to handle the fact that index is empty
    with pytest.raises(ValueError, match="No valid chunks"):
        sparse_retriever.search(request)

def test_search_request_compatibility():
    # Verify we can instantiate SearchRequest with mixed args
    req = SearchRequest(query="foo", document_id="bar", rebuild_if_stale=True)
    assert req.rebuild_if_stale is True
    assert req.source_types is None
    
    req2 = SearchRequest(query="foo", document_id="bar")
    assert req2.rebuild_if_stale is None
