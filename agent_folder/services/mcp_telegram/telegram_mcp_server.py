import os, json
from datetime import datetime, timezone
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
import uvicorn

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
HTTP_PATH = os.getenv("HTTP_PATH")

BASE = f"https://api.telegram.org/bot{TOKEN}/"
mcp = FastMCP("telegram-mcp")

def _post(method: str, data: dict) -> dict:
    with httpx.Client(timeout=35) as c:
        r = c.post(f"{BASE}{method}", data=data)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(json.dumps(j))
        return j["result"]

@mcp.tool()
def get_me() -> dict:
    return _post("getMe", {})

@mcp.tool()
def send_message(chat_id: str, text: str, parse_mode: str | None = None,
                 disable_web_page_preview: bool | None = None,
                 reply_to_message_id: int | None = None) -> dict:
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode is not None:
        payload["parse_mode"] = parse_mode
    if disable_web_page_preview is not None:
        payload["disable_web_page_preview"] = disable_web_page_preview
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    res = _post("sendMessage", payload)
    return {"message_id": res["message_id"]}

@mcp.tool()
def get_updates(offset: int | None = None, timeout: int = 20) -> dict:
    payload = {"timeout": timeout}
    if offset is not None:
        payload["offset"] = offset
    result = _post("getUpdates", payload)

    updates = []
    max_uid = None

    for u in result:
        uid = u["update_id"]
        msg = u.get("message")
        if msg and "text" in msg:
            ts = datetime.fromtimestamp(msg["date"], tz=timezone.utc).isoformat().replace("+00:00", "Z")
            updates.append({
                "update_id": uid,
                "chat_id": str(msg["chat"]["id"]),
                "user_id": str(msg.get("from", {}).get("id")) if msg.get("from") else None,
                "text": msg["text"],
                "ts": ts
            })
        max_uid = uid if max_uid is None else max(max_uid, uid)

    next_offset = (max_uid + 1) if max_uid is not None else None
    return {"updates": updates, "last_offset": next_offset}

if __name__ == "__main__":
    mcp_app = mcp.http_app(path=HTTP_PATH)
    print(f"[telegram-mcp] HTTP on http://{HOST}:{PORT}{HTTP_PATH}")
    uvicorn.run(mcp_app, host=HOST, port=PORT)
