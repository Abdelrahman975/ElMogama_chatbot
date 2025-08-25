import regex as re
from langchain.schema import Document


def normalize_text(text: str) -> str:
    normalized = str(text).strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    normalized = normalized.replace('ى', 'ي')
    normalized = re.sub(r'[\u064B-\u0652]', '', normalized)
    normalized = re.sub(r'[^\p{Arabic}\s\d.,!?;:]', '', normalized, flags=re.U)
    return normalized


def make_cache_doc(question: str, answer: str) -> Document:
    return Document(page_content=question, metadata={"answer": answer})