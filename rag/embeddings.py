from sentence_transformers import SentenceTransformer

# Load model once (global)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Convert text → vector embedding
    """
    return model.encode(text)


def embed_batch(texts: list):
    """
    Convert list of texts → list of embeddings
    """
    return model.encode(texts)