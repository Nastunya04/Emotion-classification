import os
import json
import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID")
ENV_LABELS = [s.strip() for s in (os.getenv("EMOTION_LABELS") or "").split(",") if s.strip()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
labels = None


class EmotionArgs(BaseModel):
    text: str = Field(...)


def resolve_labels():
    #will change this function, since for now I do not have label names in the config of the model
    #and need to resolve this manually
    global labels
    if labels is not None:
        return labels

    cfg = model.config
    num = getattr(cfg, "num_labels", None)

    if not num:
        if isinstance(getattr(cfg, "id2label", None), dict):
            num = len(cfg.id2label)
        elif isinstance(getattr(cfg, "label2id", None), dict):
            try:
                num = 1 + max(cfg.label2id.values())
            except Exception:
                num = len(cfg.label2id) or 0

    if ENV_LABELS and num and len(ENV_LABELS) == num:
        labels = ENV_LABELS
        return labels

    id2label = getattr(cfg, "id2label", None)
    if isinstance(id2label, dict) and num:
        names = []
        for i in range(num):
            name = id2label.get(i) or id2label.get(str(i))
            if name is None:
                label2id = getattr(cfg, "label2id", None)
                if isinstance(label2id, dict):
                    for k, v in label2id.items():
                        if v == i:
                            name = k
                            break
            names.append(name if name is not None else f"LABEL_{i}")
        labels = names
        return labels

    label2id = getattr(cfg, "label2id", None)
    if isinstance(label2id, dict) and num:
        inv = {v: k for k, v in label2id.items()}
        labels = [inv.get(i, f"LABEL_{i}") for i in range(num)]
        return labels

    labels = [f"LABEL_{i}" for i in range(num or 0)]
    return labels


def load_from_hf():
    global model, tokenizer, labels
    if model is not None:
        return
    if not HF_REPO_ID:
        raise EnvironmentError("HF_REPO_ID is not set")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)
    model.to(device).eval()
    labels = None
    resolve_labels()


def classify_emotion(text: str) -> dict:
    load_from_hf()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    resolved = resolve_labels()
    dist = {resolved[i]: float(probs[i]) for i in range(len(probs))}
    top = max(dist, key=dist.get)
    return {"top_label": top, "confidence": dist[top], "probs": dist}

FORMAT_PROMPT = ChatPromptTemplate.from_template(
    "Summarize this emotion classification for a user.\n"
    "Label: {label}\nConfidence: {confidence}\n"
    "Return one short sentence."
)

def tool_fn(text: str) -> str:
    res = classify_emotion(text)  # {"top_label": ..., "confidence": ..., "probs": {...}}
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL","gpt-4o-mini"), temperature=0)
    msg = FORMAT_PROMPT.invoke({"label": res["top_label"], "confidence": res["confidence"]})
    return llm.invoke(msg).content


emotion_tool = StructuredTool.from_function(
    name="emotion_classifier",
    description=(
        "Useful when the user wants to analyze the emotion, sentiment, feeling, or tone "
        "of a provided text snippet. Input must be the snippet itself "
        "(e.g., 'I love my mom'). Do NOT use for student facts, biography, or achievements requests."
    ),
    func=tool_fn,
    args_schema=EmotionArgs,
)
