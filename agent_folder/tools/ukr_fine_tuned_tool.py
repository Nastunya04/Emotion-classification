import os, json, torch, requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

SERVICE_MODE: str = (os.getenv("SERVICE_MODE") or "local").lower()
EMOTION_URL_UKR: Optional[str] = os.getenv("EMOTION_URL_UKR")
LLM_MODEL: str = os.getenv("LLM_MODEL")
UKR_MODEL: Optional[str] = os.getenv("UKR_MODEL")

ID2LABEL = {
    0: "Joy",
    1: "Fear",
    2: "Anger",
    3: "Sadness",
    4: "Disgust",
    5: "Surprise"
}

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
    if not UKR_MODEL:
        raise EnvironmentError("UKR_MODEL is not set")
    print(f"[emotion_ukr] Loading local model from {UKR_MODEL}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(UKR_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(UKR_MODEL).to(device).eval()
    cfg = _model.config
    num = getattr(cfg, "num_labels", None)

    if num is not None and num == len(ID2LABEL):
        _labels = [ID2LABEL[i] for i in range(num)]
    else:
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and num is not None:
            _labels = [id2label.get(i) or id2label.get(str(i)) or f"LABEL_{i}" for i in range(num)]
        else:
            _labels = [f"LABEL_{i}" for i in range(num or 0)]
    print(f"[emotion_ukr] Model loaded. Labels: {_labels}")

def classify_local_ukr(text: str) -> Dict[str, Any]:
    print(f"[emotion_ukr] classify_local called with text='{text}'")
    _load_local_model()
    device = next(_model.parameters()).device
    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = _model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    dist = {_labels[i]: float(probs[i]) for i in range(len(probs))}
    top = max(dist, key=dist.get)
    print(f"[emotion_ukr] classify_local result: top={top}, confidence={dist[top]:.3f}")
    return {"top_label": top, "confidence": dist[top], "probs": dist}

def classify_remote_ukr(text: str) -> Dict[str, Any]:
    if not EMOTION_URL_UKR:
        raise EnvironmentError("EMOTION_URL_UKR is not set")
    print(f"[emotion_ukr] classify_remote POST to {EMOTION_URL_UKR} with text='{text}'")
    r = requests.post(EMOTION_URL_UKR, json={"text": text}, timeout=1000)
    print(f"[emotion_ukr] classify_remote response status={r.status_code}")
    r.raise_for_status()
    data = r.json()
    print(f"[emotion_ukr] classify_remote result: {data}")
    return {
        "top_label": data["top_label"],
        "confidence": float(data["confidence"]),
        "probs": data["probs"],
    }

def classify_emotion_ukr(text: str) -> Dict[str, Any]:
    print(f"[emotion_ukr] classify_emotion SERVICE_MODE={SERVICE_MODE}")
    return classify_remote_ukr(text) if SERVICE_MODE == "remote" else classify_local_ukr(text)

def tool_fn(text: str) -> str:
    print(f"[emotion_ukr] tool_fn invoked with text='{text}'")
    res = classify_emotion_ukr(text)
    if os.getenv("OPENAI_API_KEY"):
        print(f"[emotion_ukr] Summarizing with LLM={LLM_MODEL}")
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        msg = FORMAT_PROMPT.invoke({"label": res["top_label"], "confidence": res["confidence"]})
        summary = llm.invoke(msg).content
    else:
        summary = f"{res['top_label']} (confidence={res['confidence']:.3f})"
    out = json.dumps({"summary": summary, **res}, ensure_ascii=False)
    print(f"[emotion_ukr] tool_fn output={out}")
    return out

ukr_emotion_tool = StructuredTool.from_function(
    name="emotion_classifier_ukr",
    description="Analyze the emotion/sentiment/tone of a Ukrainian text snippet. Use for Ukrainian inputs.",
    func=tool_fn,
    args_schema=EmotionArgs,
)