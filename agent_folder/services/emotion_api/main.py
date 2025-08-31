from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os

from transformers.utils.logging import enable_progress_bar
from tools.fine_tuned_tool import classify_emotion, FORMAT_PROMPT
from tools.ukr_fine_tuned_tool import classify_emotion_ukr 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

enable_progress_bar()
load_dotenv()

app = FastAPI(title="Emotion API")

class ClassifyIn(BaseModel):
    text: str

class ClassifyOut(BaseModel):
    summary: str
    top_label: str
    confidence: float
    probs: Dict[str, float]

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyOut)
def classify(payload: ClassifyIn) -> Any:
    try:
        res = classify_emotion(payload.text)
        summary = f"{res['top_label']} (confidence={res['confidence']:.3f})"

        llm_model = os.getenv("LLM_MODEL")
        if llm_model and os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model=llm_model, temperature=0)
            msg = FORMAT_PROMPT.invoke(
                {"label": res["top_label"], "confidence": res["confidence"]}
            )
            summary = llm.invoke(msg).content

        return {
            "summary": summary,
            "top_label": res["top_label"],
            "confidence": float(res["confidence"]),
            "probs": {k: float(v) for k, v in res["probs"].items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_ukr", response_model=ClassifyOut)
def classify_ukr(payload: ClassifyIn) -> Any:
    try:
        res = classify_emotion_ukr(payload.text)
        summary = f"{res['top_label']} (confidence={res['confidence']:.3f})"
        llm_model = os.getenv("LLM_MODEL")
        if llm_model and os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model=llm_model, temperature=0)
            msg = FORMAT_PROMPT.invoke({"label": res["top_label"], "confidence": res["confidence"]})
            summary = llm.invoke(msg).content
        return {
            "summary": summary,
            "top_label": res["top_label"],
            "confidence": float(res["confidence"]),
            "probs": {k: float(v) for k, v in res["probs"].items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))