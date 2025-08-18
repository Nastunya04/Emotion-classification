import os
import json
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from dotenv import load_dotenv
load_dotenv()

VECTOR_DIR = os.getenv("VECTOR_DIR", ".faiss_vs")
INDEX_NAME = os.getenv("INDEX_NAME", "student_index")
INDEX_PATH = os.path.join(VECTOR_DIR, INDEX_NAME)
PROVIDER = (os.getenv("EMBEDDINGS_PROVIDER") or "openai").lower()
EMB_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

class RAGArgs(BaseModel):
    question: str = Field(...)

def emb():
    if PROVIDER == "openai":
        return OpenAIEmbeddings(model=EMB_MODEL)
    hf_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=hf_model)

def base_retriever():
    store = FAISS.load_local(INDEX_PATH, emb(), allow_dangerous_deserialization=True)
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
    )

def compressed_retriever():
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=llm_model, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever(),
    )

PROMPT = ChatPromptTemplate.from_template(
    "Use ONLY the provided context. Do NOT invent information.\n\n"
    "If the context is rich, answer concisely.\n"
    "If the context is sparse (few details available), explicitly acknowledge it by starting with:\n"
    "'From the context, I only know that...' or 'I do not have much information, only that...'\n"
    "Then list the available facts.\n\n"
    "Question: {q}\n"
    "Context:\n{ctx}\n\n"
    "Final Answer:"
)

def rag_answer(question: str) -> dict:
    if not os.path.exists(INDEX_PATH) and not os.path.exists(INDEX_PATH + ".faiss"):
        return {"answer": "RAG index not found. Please run ingestion.", "sources": []}

    retriever = compressed_retriever()
    docs = retriever.invoke(question)

    if len(docs) < 2:
        base = base_retriever()
        docs = base.invoke(question)

    chunks, sources = [], []
    for d in docs:
        chunks.append(d.page_content)
        sources.append({"metadata": d.metadata})

    ctx = "\n\n".join(chunks)

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=llm_model, temperature=0)
    msg = PROMPT.invoke({"q": question, "ctx": ctx})
    out = llm.invoke(msg)

    return {
        "answer": out.content,
        "sources": sources,
    }

def tool_fn(question: str) -> str:
    return json.dumps(rag_answer(question), ensure_ascii=False)

student_rag_tool = StructuredTool.from_function(
    name="student_rag",
    description="Useful when the user has factual questions about the student, "
                "such as birth date, university, program, projects, achievements, languages, etc. "
                "Answer strictly from retrieved context. "
                "Do NOT use for emotion/sentiment analysis.",
    func=tool_fn,
    args_schema=RAGArgs,
)
