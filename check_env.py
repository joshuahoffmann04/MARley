try:
    import rank_bm25
    print("rank_bm25: OK")
except ImportError as e:
    print(f"rank_bm25: FAIL {e}")

try:
    import chromadb
    print("chromadb: OK")
except ImportError as e:
    print(f"chromadb: FAIL {e}")

try:
    import tiktoken
    print("tiktoken: OK")
except ImportError as e:
    print(f"tiktoken: FAIL {e}")

from retrieval.sparse_retrieval.service import SparseBM25Retriever
print("SparseBM25Retriever import: OK")
