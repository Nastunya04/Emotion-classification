from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from tools.rag_tool import rag_answer
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG API")

class AskIn(BaseModel):
    question: str

class AskOut(BaseModel):
    answer: str

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn) -> Any:
    try:
        out = rag_answer(payload.question)
        return {"answer": out.get("answer", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))