from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from agent.agent import build_agent_with_history, get_session_history

from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Agent API")
agent_with_history = build_agent_with_history()

class ChatIn(BaseModel):
    text: str
    session_id: Optional[str] = "default"

class ChatOut(BaseModel):
    output: str
    session_id: str
    history_len: int

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn) -> Any:
    try:
        session_id = payload.session_id or "default"
        cfg = {"configurable": {"session_id": session_id}}
        out = agent_with_history.invoke({"input": payload.text}, config=cfg)
        hist = get_session_history(session_id).messages
        return {
            "output": out.get("output", ""),
            "session_id": session_id,
            "history_len": len(hist),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))