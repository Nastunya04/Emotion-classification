import os
import json
from typing import Dict, List, Any, Optional

import requests
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

SERVICE_MODE: str = (os.getenv("SERVICE_MODE") or "local").lower()
RAG_URL: Optional[str] = os.getenv("RAG_URL")
LLM_MODEL: str = os.getenv("LLM_MODEL")
VECTOR_DIR: str = os.getenv("VECTOR_DIR")
INDEX_NAME: str = os.getenv("INDEX_NAME")
INDEX_PATH: str = os.path.join(VECTOR_DIR, INDEX_NAME)
PROVIDER: str = (os.getenv("EMBEDDINGS_PROVIDER") or "openai").lower()
EMB_MODEL: Optional[str] = os.getenv("EMBEDDINGS_MODEL")
HF_EMBED_MODEL: str = os.getenv("HF_EMBED_MODEL")

PROMPT = ChatPromptTemplate.from_template(
    "Use ONLY the provided context. Do NOT invent information.\n\n"
    "If the context is rich, answer concisely.\n"
    "If the context is sparse, start with one of:\n"
    "'From the context, I only know that...' or 'I do not have much information, only that...'\n"
    "Then list the available facts.\n\n"
    "Question: {q}\n"
    "Context:\n{ctx}\n\n"
    "Final Answer:"
)

class RAGArgs(BaseModel):
    question: str = Field(..., description="The user's question about the student.")

def _embeddings():
    if PROVIDER == "openai":
        return OpenAIEmbeddings(model=EMB_MODEL)
    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

def _index_present() -> bool:
    if not os.path.isdir(INDEX_PATH):
        return False
    return (
        os.path.exists(os.path.join(INDEX_PATH, "index.faiss"))
        and os.path.exists(os.path.join(INDEX_PATH, "index.pkl"))
    )

def rag_local(question: str) -> Dict[str, Any]:
    if not _index_present():
        return {"answer": "RAG index not found. Please run ingestion.", "sources": []}

    store = FAISS.load_local(
        INDEX_PATH,
        _embeddings(),
        allow_dangerous_deserialization=True,
    )

    base = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base
    )

    docs = retriever.invoke(question) or []
    if len(docs) < 2:
        docs = base.invoke(question) or []

    chunks: List[str] = []
    sources: List[Dict[str, Any]] = []
    for d in docs:
        chunks.append(d.page_content)
        src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
        sources.append({"source": src})

    ctx = "\n\n".join(chunks)
    msg = PROMPT.invoke({"q": question, "ctx": ctx})
    out = llm.invoke(msg)

    return {"answer": out.content, "sources": sources}

def rag_remote(question: str) -> Dict[str, Any]:
    if not RAG_URL:
        return {"answer": "RAG_URL is not configured.", "sources": []}
    resp = requests.post(RAG_URL, json={"question": question}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return {"answer": data.get("answer", ""), "sources": data.get("sources", [])}

def rag_answer(question: str) -> Dict[str, Any]:
    return rag_remote(question) if SERVICE_MODE == "remote" else rag_local(question)

def tool_fn(question: str) -> str:
    return json.dumps(rag_answer(question), ensure_ascii=False)

student_rag_tool = StructuredTool.from_function(
    name="student_rag",
    description="Answer factual questions about the student strictly from retrieved context.",
    func=tool_fn,
    args_schema=RAGArgs,
)
