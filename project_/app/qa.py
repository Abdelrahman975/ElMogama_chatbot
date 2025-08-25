from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from .retriever import retriever_manager
from .prompt_template import PROMPT
from .config import OLLAMA_MODEL

qa_chain = None


def init_qa_chain():
    global qa_chain
    llm = OllamaLLM(model=OLLAMA_MODEL)  # allow the constructor to accept environment/config-based model
    # use the main vector DB as retriever (not the small semantic cache)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever_manager.vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain