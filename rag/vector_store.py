import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from rag.embeddings import embed_text, embed_batch

# In-memory stores
dimension = 384  # for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

documents = []  # store original text chunks
bm25 = None    # for keyword search

def reset_knowledge_base():
    """
    Clears the entire knowledge base for a fresh reload
    """
    global documents, bm25, index
    documents = []
    bm25 = None
    index = faiss.IndexFlatL2(dimension)

def add_documents(text_chunks: list):
    """
    Add documents to both FAISS and BM25 index
    """
    global documents, bm25

    # 1. Vector Search setup
    embeddings = embed_batch(text_chunks)
    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)

    # 2. Update document list
    documents.extend(text_chunks)
    
    # 3. Keyword Search (BM25) setup - rebuild on update
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)


def search(query: str, top_k: int = 3):
    """
    Hybrid Search: Semantic (FAISS) + Keyword (BM25)
    Filters out semantically irrelevant chunks using a distance threshold.
    """
    if not documents:
        return []

    # 1. Semantic Search with distance threshold
    query_vector = embed_text(query)
    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    # Filter: only keep results with L2 distance < 1.2 (tune as needed)
    DISTANCE_THRESHOLD = 1.2
    semantic_results = [
        documents[idx]
        for dist, idx in zip(distances[0], indices[0])
        if idx < len(documents) and dist < DISTANCE_THRESHOLD
    ]

    if not semantic_results:
        return []

    # 2. Keyword Search (BM25) - only if bm25 is initialized
    if bm25:
        query_tokens = query.lower().split()
        bm25_results = bm25.get_top_n(query_tokens, documents, n=top_k)
    else:
        bm25_results = []

    # 3. Simple Fusion (Combine and deduplicate)
    combined = list(dict.fromkeys(semantic_results + bm25_results))

    return combined[:top_k]