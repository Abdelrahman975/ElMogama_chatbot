import os
import faiss
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from .config import CSV_PATH, VECTOR_DB_PATH, CACHE_DB_PATH, CACHE_SIMILARITY_THRESHOLD
from .embeddings import get_embeddings
import pandas as pd
from .utils import normalize_text


class RetrieverManager:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_db: Optional[FAISS] = None
        self.cache_db: Optional[FAISS] = None
        self.cached_documents: List[Document] = []

    def init_vector_db(self):
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_db = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        else:
            if not os.path.exists(CSV_PATH):
                raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
            df = pd.read_csv(CSV_PATH).dropna().drop_duplicates()
            texts = [f"السؤال: {normalize_text(row.iloc[0])}\nالإجابة: {str(row.iloc[1])}" for _, row in df.iterrows()]
            self.vector_db = FAISS.from_texts(texts, self.embeddings)
            self.vector_db.save_local(VECTOR_DB_PATH)

    def init_cache_db(self):
        # Load cache if exists, otherwise create empty index with correct dim
        if os.path.exists(CACHE_DB_PATH):
            try:
                self.cache_db = FAISS.load_local(CACHE_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                # try to extract docstore docs if available
                try:
                    ids = getattr(self.cache_db, "index_to_docstore_id", [])
                    self.cached_documents = [self.cache_db.docstore.get_document(doc_id) for doc_id in ids]
                except Exception:
                    self.cached_documents = []
            except Exception:
                self.cached_documents = []
        else:
            # create an empty faiss index with the right dim
            dummy_emb = self.embeddings.embed_query("temp_dummy")
            dim = len(dummy_emb)
            index = faiss.IndexFlatL2(dim)
            self.cache_db = FAISS(self.embeddings.embed_query, index, {}, {})
            # do not save yet until there is at least one doc
            self.cached_documents = []

    def save_cache(self, path=CACHE_DB_PATH):
        if not self.cached_documents:
            # Save empty index to disk to keep shape
            dummy_emb = self.embeddings.embed_query("temp_dummy")
            dim = len(dummy_emb)
            index = faiss.IndexFlatL2(dim)
            temp_db = FAISS(self.embeddings.embed_query, index, {}, {})
            temp_db.save_local(path)
        else:
            temp_db = FAISS.from_documents(self.cached_documents, self.embeddings)
            temp_db.save_local(path)

    def query_cache(self, query_embedding, k=1) -> Optional[Tuple[Document, float]]:
        if self.cache_db is None or getattr(self.cache_db, 'index', None) is None:
            return None
        if getattr(self.cache_db.index, 'ntotal', 0) == 0:
            return None
        results = self.cache_db.similarity_search_with_score_by_vector(query_embedding, k=k)
        if results:
            doc, score = results[0]
            # return only if it meets threshold
            if score < CACHE_SIMILARITY_THRESHOLD:
                return doc, score
        return None

    def add_to_cache(self, doc):
        # keep in memory list, actual FAISS rebuild/save happens separately
        self.cached_documents.append(doc)


retriever_manager = RetrieverManager()
