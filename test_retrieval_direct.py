import json
from pathlib import Path
from tempfile import TemporaryDirectory
from retrieval.base import SearchRequest
from retrieval.sparse_retrieval.service import SparseBM25Retriever
from retrieval.sparse_retrieval.config import SparseRetrievalSettings, get_sparse_retrieval_config

def main():
    print("Setting up temporary directory...")
    with TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str)
        doc_id = "test_doc"
        doc_dir = tmp_path / doc_id
        chunks_dir = doc_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        print(f"Chunks dir: {chunks_dir}")
        
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
        print(f"Created chunk file: {chunk_file}")
        
        settings = SparseRetrievalSettings(
            document_id=doc_id,
            data_root=tmp_path,
            app_name="TestApp"
        )
        print(f"Settings data_root: {settings.data_root}")
        print(f"Settings document_root: {settings.document_root}")
        print(f"Settings input_dir_path: {settings.input_dir_path}")
        
        config = get_sparse_retrieval_config(settings)
        print(f"Config input_dir: {config.input_dir}")
        
        retriever = SparseBM25Retriever(settings=settings, runtime_config=config)
        
        req = SearchRequest(
            query="retrieval",
            document_id=doc_id, # Must match!
            rebuild_if_stale=True
        )
        
        print("Calling retriever.search...")
        
        # Check tokenization first
        tokens = retriever._tokenize("retrieval")
        print(f"Query tokens: {tokens}")
        
        try:
            resp = retriever.search(req)
            print("Search successful!")
            print(f"Hits: {len(resp.hits)}")
            for h in resp.hits:
                print(f"Hit: {h.chunk_id}, Score: {h.score}, Content: {h.content}")
                
            # Inspect state
            state = retriever._state
            if state:
                print(f"State doc count: {len(state.documents)}")
                if state.documents:
                    print(f"Doc 0 tokens: {state.documents[0].tokens}")
                    print(f"Doc 0 text: {state.documents[0].text}")
                    scores = state.bm25.get_scores(tokens)
                    print(f"Scores for query tokens: {scores}")
            
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
