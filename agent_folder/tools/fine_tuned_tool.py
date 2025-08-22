import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

load_dotenv()

SERVICE_MODE: str = (os.getenv("SERVICE_MODE") or "local").lower()
EMOTION_URL: Optional[str] = os.getenv("EMOTION_URL")
LLM_MODEL: str = os.getenv("LLM_MODEL")
HF_REPO_ID: Optional[str] = os.getenv("HF_REPO_ID")
ENV_LABELS = [s.strip() for s in (os.getenv("EMOTION_LABELS") or "").split(",") if s.strip()]

FORMAT_PROMPT = ChatPromptTemplate.from_template(
    "Summarize this emotion classification for a user.\nLabel: {label}\nConfidence: {confidence}\nReturn one short sentence."
)

class EmotionArgs(BaseModel):
    text: str = Field(...)

_tokenizer = None
_model = None
_labels = None

def _load_local_model():
    global _tokenizer, _model, _labels
    if _model is not None:
        return
    if not HF_REPO_ID:
        raise EnvironmentError("HF_REPO_ID is not set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID).to(device).eval()
    cfg = _model.config
    num = getattr(cfg, "num_labels", None)
    if ENV_LABELS and num and len(ENV_LABELS) == num:
        _labels = ENV_LABELS
    else:
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and num is not None:
            _labels = [id2label.get(i) or id2label.get(str(i)) or f"LABEL_{i}" for i in range(num)]
        else:
            _labels = [f"LABEL_{i}" for i in range(num or 0)]

def classify_local(text: str) -> Dict[str, Any]:
    _load_local_model()
    device = next(_model.parameters()).device
    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = _model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    dist = {_labels[i]: float(probs[i]) for i in range(len(probs))}
    top = max(dist, key=dist.get)
    return {"top_label": top, "confidence": dist[top], "probs": dist}

def classify_remote(text: str) -> Dict[str, Any]:
    if not EMOTION_URL:
        raise EnvironmentError("EMOTION_URL is not set")
    r = requests.post(EMOTION_URL, json={"text": text}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        "top_label": data["top_label"],
        "confidence": float(data["confidence"]),
        "probs": data["probs"],
    }

def classify_emotion(text: str) -> Dict[str, Any]:
    return classify_remote(text) if SERVICE_MODE == "remote" else classify_local(text)

def tool_fn(text: str) -> str:
    res = classify_emotion(text)
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        msg = FORMAT_PROMPT.invoke({"label": res["top_label"], "confidence": res["confidence"]})
        summary = llm.invoke(msg).content
    else:
        summary = f"{res['top_label']} (confidence={res['confidence']:.3f})"
    return json.dumps({"summary": summary, **res}, ensure_ascii=False)

emotion_tool = StructuredTool.from_function(
    name="emotion_classifier",
    description="Analyze the emotion/sentiment/tone of a text snippet. Use for emotional content, not student facts.",
    func=tool_fn,
    args_schema=EmotionArgs,
)
