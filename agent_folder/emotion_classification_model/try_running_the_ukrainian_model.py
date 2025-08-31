import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()
id2label = {
    0: "Joy",
    1: "Fear",
    2: "Anger",
    3: "Sadness",
    4: "Disgust",
    5: "Surprise"
}

MODEL_ID = os.getenv("UKR_MODEL")
if not MODEL_ID:
    raise ValueError("Environment variable UKR_MODEL is not set")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

@torch.no_grad()
def predict(texts, topk=3, max_len=256):
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    top_ids = np.argsort(-probs, axis=1)[:, :topk]
    results = []
    for i, text in enumerate(texts):
        preds = [(id2label[idx], float(probs[i, idx])) for idx in top_ids[i]]
        results.append({"text": text, "predictions": preds})
    return results

samples = [
    "Я така щаслива сьогодні!",
    "Мені страшно виходити на сцену.",
    "Мене це дратує.",
    "Я люблю цю пісню.",
    "Мені дуже сумно.",
    "Ого, зовсім не очікувала такого результату!"
]

out = predict(samples, topk=3)
for r in out:
    print(r["text"])
    for lbl, score in r["predictions"]:
        print(f"  {lbl}: {score:.4f}")
    print()