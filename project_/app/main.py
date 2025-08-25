import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import QuestionRequest
from .config import CACHE_FLUSH_THRESHOLD, CACHE_DB_PATH
from .embeddings import get_embeddings
from .retriever import retriever_manager
from .qa import init_qa_chain
from .utils import make_cache_doc

app = FastAPI(title="Arabic FAQ Chatbot API")

# Allow CORS in case you host a frontend (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")

qa_chain = None
_cached_additions = 0


@app.on_event("startup")
async def startup_event():
    global qa_chain
    logger.info("Starting up: initializing embeddings, retrievers and QA chain...")

    # init embeddings via retriever (they share same embeddings instance)
    retriever_manager.init_vector_db()
    retriever_manager.init_cache_db()

    # init qa chain
    qa_chain = init_qa_chain()
    logger.info("Startup finished.")


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    global _cached_additions
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    start = time.time()
    try:
        embeddings = get_embeddings()
        normalized_question = embeddings._normalize_text(req.question) if hasattr(embeddings, '_normalize_text') else req.question
        query_emb = embeddings.embed_query(normalized_question)

        # 1. check semantic cache
        cache_hit = retriever_manager.query_cache(query_emb, k=1)
        if cache_hit is not None:
            doc, score = cache_hit
            elapsed = time.time() - start
            return {"answer": doc.metadata.get("answer", ""), "cached": True, "elapsed_seconds": round(elapsed, 2)}

        # 2. call LLM
        logger.info("Cache miss — calling LLM")
        response = qa_chain.invoke({"query": req.question})
        raw_answer = response.get("result") if isinstance(response, dict) else str(response)
        answer = raw_answer

        # 3. clean answer and add to cache memory
        # remove think tags if present
        import re as _re
        answer = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL).strip()

        doc = make_cache_doc(normalized_question, answer)
        retriever_manager.add_to_cache(doc)
        _cached_additions += 1

        # flush to disk periodically
        if _cached_additions >= CACHE_FLUSH_THRESHOLD:
            retriever_manager.save_cache(CACHE_DB_PATH)
            # reload cache_db so it reflects the saved index
            retriever_manager.init_cache_db()
            _cached_additions = 0

        elapsed = time.time() - start
        return {"answer": answer, "cached": False, "elapsed_seconds": round(elapsed, 2)}

    except Exception as e:
        logger.exception("Error processing question")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reload_cache")
async def reload_cache():
    try:
        retriever_manager.save_cache(CACHE_DB_PATH)
        retriever_manager.init_cache_db()
        return {"status": "cache reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}