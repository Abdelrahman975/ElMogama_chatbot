import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "faq.csv"
VECTOR_DB_PATH = DATA_DIR / "faiss_index"
CACHE_DB_PATH = DATA_DIR / "faiss_cache_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "qwen3:1.7b"
CACHE_SIMILARITY_THRESHOLD = 0.45
CACHE_FLUSH_THRESHOLD = 10  # save cache to disk every N additions
EMBEDDING_CACHE_FOLDER = BASE_DIR / ".embedding_cache"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)