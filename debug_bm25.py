from rank_bm25 import BM25Okapi

corpus = [
    ['this', 'is', 'a', 'test', 'chunk', 'for', 'retrieval'],
    ['just', 'another', 'chunk', 'without', 'the', 'keyword']
]

bm25 = BM25Okapi(
    corpus=corpus,
    k1=1.5,
    b=0.75,
    epsilon=0.25
)

query = ['retrieval']
scores = bm25.get_scores(query)
print(f"Index 0 (doc with term) score: {scores[0]}")
print(f"Index 1 (doc without term) score: {scores[1]}")
print(f"All scores: {scores}")

# Check IDF
# Accessing internal idf if possible
# Usually bm25.idf is a dict
if hasattr(bm25, 'idf'):
    print(f"IDF for 'retrieval': {bm25.idf.get('retrieval')}")
