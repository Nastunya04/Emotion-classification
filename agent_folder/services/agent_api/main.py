import logging, traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from agent.agent import build_agent_with_history, get_session_history, clear_session_history
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Agent API")
agent_with_history = build_agent_with_history()

logger = logging.getLogger("agent-api")
logging.basicConfig(level=logging.INFO)

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
        tb = traceback.format_exc()
        logger.error(f"[chat] error for session={payload.session_id} text='{payload.text}': {e}\n{tb}")
        raise HTTPException(status_code=500, detail="Agent internal error")

@app.get("/mem")
def mem(session_id: str = "default") -> Any:
    try:
        hist = get_session_history(session_id).messages
        return [getattr(m, "content", str(m)) for m in hist]
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[mem] error for session={session_id}: {e}\n{tb}")
        raise HTTPException(status_code=500, detail="Agent internal error")

class ClearIn(BaseModel):
    session_id: Optional[str] = "default"

@app.post("/clear")
def clear(payload: ClearIn) -> Dict[str, str]:
    try:
        session_id = payload.session_id or "default"
        clear_session_history(session_id)
        return {"status": "ok"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[clear] error for session={payload.session_id}: {e}\n{tb}")
        raise HTTPException(status_code=500, detail="Agent internal error")