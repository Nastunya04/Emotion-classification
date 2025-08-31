import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from dotenv import load_dotenv

load_dotenv()
repo_id = os.getenv("HF_REPO_ID")
if repo_id is None:
    raise EnvironmentError("HF_REPO_ID environment variable is not set. Please define it in your .env or shell.")

tokenizer = AutoTokenizer.from_pretrained(repo_id)
config = AutoConfig.from_pretrained(repo_id)

id2label = config.id2label
label2id = config.label2id
labels = list(label2id.keys())

model = AutoModelForSequenceClassification.from_pretrained(repo_id, config=config)
model.eval()

text = "I love my mom"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits[0]
probs = torch.softmax(logits, dim=-1).tolist()

labels_map = model.config.id2label
dist = {labels_map[i]: float(p) for i, p in enumerate(probs)}
top = max(dist, key=dist.get)

print("Distribution:", json.dumps(dist, indent=2))
print(f"Top label: {top} (p={dist[top]:.4f})")