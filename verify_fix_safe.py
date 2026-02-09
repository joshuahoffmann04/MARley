import json
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from retrieval.base import SearchRequest
from retrieval.sparse_retrieval.service import SparseBM25Retriever
from retrieval.sparse_retrieval.config import SparseRetrievalSettings, get_sparse_retrieval_config
from pydantic import ValidationError

def main():
    try:
        with TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            doc_id = "test_doc"
            doc_dir = tmp_path / doc_id
            chunks_dir = doc_dir / "chunks"
            chunks_dir.mkdir(parents=True)
            
            chunk_file = chunks_dir / f"{doc_id}-pdf-chunker-output.json"
            # 3 chunks to ensure N=3, n=1 => IDF > 0
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
            
            settings = SparseRetrievalSettings(
                document_id=doc_id,
                data_root=tmp_path,
                app_name="TestApp"
            )
            
            config = get_sparse_retrieval_config(settings)
            retriever = SparseBM25Retriever(settings=settings, runtime_config=config)
            
            req = SearchRequest(
                query="retrieval",
                document_id=doc_id,
                rebuild_if_stale=True
            )
            
            resp = retriever.search(req)
            if len(resp.hits) == 1 and resp.hits[0].chunk_id == "c1":
                print("PASS")
            else:
                print(f"FAIL: Hits={len(resp.hits)}")
                for h in resp.hits:
                    print(f"Hit: {h.chunk_id} Score: {h.score}")

    except ValidationError as e:
        print("Validation Error:")
        print(e.json(indent=2))
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
