# Agent for Emotion Classification & Student RAG

A unified agent that combines **emotion classification** and **retrieval-augmented generation**, exposed as FastAPI microservices and integrated with Telegram via MCP.

A unified agent that combines emotion classification and retrieval-augmented generation, exposed as FastAPI microservices and integrated with Telegram via MCP.

- **English Emotion Classifier** — a fine-tuned BERT model for 6 emotions: *joy, sadness, anger, fear, love, surprise*.  
  Weights are available on Hugging Face: [nastiadynia/emot_classification](https://huggingface.co/nastiadynia/emot_classification)
- **Ukrainian Emotion Classifier (bonus)** — an additional XLM-R model trained on a Ukrainian emotion dataset for 6 emotions: *joy, fear, anger, sadness, disgust, surprise*.  
  Weights are available on Hugging Face: [nastiadynia/ukrainian_emot_classification](https://huggingface.co/nastiadynia/ukrainian_emot_classification)
- **RAG Component** — builds a FAISS index from `data/student_profile/` and answers factual questions about the student using only that text.
- **Unified Agent** — LangChain tool-selection between EN/UA classifiers and RAG. Deployed as modular services: `agent-api`, `emotion-api`, `rag-api`, plus Telegram bridge via MCP.

## Project Structure
```
agent_folder/
├── agent/                     # Core agent logic (LangChain tools, CLI)
│   ├── agent.py               # Builds agent with tool-selection rules + memory
│   └── cli.py                 # CLI interface for the agent
│
├── data/student_profile/      # Source texts for RAG (student biography)
│   └── profile.txt
│
├── emotion_classification_model/   # Standalone test scripts for models
│   ├── try_running_the_english_model.py
│   └── try_running_the_ukrainian_model.py
│
├── rag/                       # RAG index builder
│   └── ingest.py              # Loads .txt files, builds FAISS index
│
├── services/                  # FastAPI microservices
│   ├── agent_api/             # Unified agent API (chat, mem, clear)
│   ├── emotion_api/           # Emotion classification API (EN + UA)
│   ├── rag_api/               # RAG API (answers factual questions)
│   ├── rag_ingest/            # Ingestion service for index build
│   └── mcp_telegram/          # Telegram bridge via MCP
│
├── tools/                     # LangChain tool wrappers
│   ├── fine_tuned_tool.py     # English classifier tool
│   ├── ukr_fine_tuned_tool.py # Ukrainian classifier tool
│   └── rag_tool.py            # RAG tool
│
├── docker-compose.yml         # Multi-service setup
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables
```
## Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- An OpenAI API key (for embeddings/LLM)
  
### 1) Clone this github
```bash
git clone <https://github.com/Nastunya04/Emotion-classification.git> && cd agent_folder
```

### 2) Create a .env in the repo root:
```
# API keys and models
OPENAI_API_KEY=...              # OpenAI key for embeddings or LLM
LLM_MODEL=gpt-4o-mini           # LLM used for summaries/tool orchestration

# Data & vector index
DATA_DIR=data/student_profile   # path to profile data
VECTOR_DIR=.faiss_vs            # where FAISS index is stored
INDEX_NAME=student_index        # name of FAISS index folder

# Embeddings provider
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-ada-002

# Emotion classification models
HF_REPO_ID=nastiadynia/emot_classification          # English emotion model
UKR_MODEL=nastiadynia/ukrainian_emot_classification # Ukrainian emotion model

# Mode: local (direct model calls) or remote (API between services)
SERVICE_MODE=remote

# Internal service URLs
EMOTION_URL=http://emotion-api:8000/classify
EMOTION_URL_UKR=http://emotion-api:8000/classify_ukr
RAG_URL=http://rag-api:8000/ask

# Telegram integration
TELEGRAM_BOT_TOKEN=...
TELEGRAM_MCP_URL=http://telegram-mcp:8081/mcp
PORT=8081
HOST=0.0.0.0
HTTP_PATH=/mcp

# Agent API endpoints
AGENT_API_URL=http://agent-api:8000
AGENT_CHAT_PATH=/chat
AGENT_MEM_PATH=/mem
AGENT_CLEAR_PATH=/clear

# Polling timeout for Telegram bridge
TELEGRAM_POLL_TIMEOUT=20
```

## How to run

### A) Docker

In this mode everything is started automatically via `docker-compose.yml`, including building the RAG index by the `rag-ingest` service. No local commands are needed.

```bash
docker compose up --build
```
After startup, the following services will be available:
- **rag-api**
  
	•	`POST` http://localhost:8002/ask — answers factual questions about the student (RAG).
- **emotion-api**
  
	•	`POST` http://localhost:8003/classify — English emotion classifier (BERT).

	•	`POST` http://localhost:8003/classify_ukr — Ukrainian emotion classifier (XLM-R).
- **agent-api**
  
	•	`POST` http://localhost:8001/chat — single entry point, agent automatically selects the right tool.

	•	`GET`  http://localhost:8001/mem — get conversation memory.

	•	`POST` http://localhost:8001/clear — clear conversation memory.

	•	`GET`  http://localhost:8001/health — health check.

If you configure `TELEGRAM_BOT_TOKEN` in `.env`, the **telegram-mcp** and **telegram-agent-bridge** services will also run. They connect `Telegram → MCP → agent-api`, so you can chat with the bot directly in Telegram.

#### Health checks
```bash
curl http://localhost:8002/health   # rag-api
curl http://localhost:8003/health   # emotion-api
curl http://localhost:8001/health   # agent-api
```

### B) Local dev
Use this mode if you want to run modules directly from Python.
#### 	1.Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2.	Build the RAG index (one-time):
```bash
python -m rag.ingest
```

#### 3. Run the CLI agent:
```bash
python -m agent.cli
# type your messages directly; /mem will show memory; exit will quit the session
```







