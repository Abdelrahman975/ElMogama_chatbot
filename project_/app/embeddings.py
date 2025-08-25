from typing import Any
from .config import EMBEDDING_MODEL, EMBEDDING_CACHE_FOLDER
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings: Any = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # instantiate once
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # cache_folder=str(EMBEDDING_CACHE_FOLDER),
        )
    return _embeddings