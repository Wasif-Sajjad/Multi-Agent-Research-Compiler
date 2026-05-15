import os

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads and returns the HuggingFace embeddings model.
    Defaults to all-MiniLM-L6-v2 if EMBEDDING_MODEL is not set in the environment.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)
