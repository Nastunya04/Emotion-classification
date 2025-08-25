import os, asyncio, json, time
from typing import Any
import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import traceback
from dotenv import load_dotenv

load_dotenv()

MCP_URL = os.getenv("TELEGRAM_MCP_URL")
AGENT_API_URL = os.getenv("AGENT_API_URL")
CHAT_PATH = os.getenv("AGENT_CHAT_PATH")
MEM_PATH = os.getenv("AGENT_MEM_PATH")
CLEAR_PATH = os.getenv("AGENT_CLEAR_PATH")
TIMEOUT = int(os.getenv("TELEGRAM_POLL_TIMEOUT"))

STOPPED = set()

def unwrap(resp: Any) -> Any:
    content = getattr(resp, "content", None)
    if not content: return resp
    item = content[0]
    t = getattr(item, "text", None)
    if isinstance(t, str):
        try:
            return json.loads(t)
        except Exception:
            return t
    md = getattr(item, "model_dump", None)
    if callable(md): return md()
    mdj = getattr(item, "model_dump_json", None)
    if callable(mdj):
        try: return json.loads(mdj())
        except Exception: return mdj()
    return item

def extract_updates(obj: Any):
    if not isinstance(obj, dict): return [], None
    return obj.get("updates") or [], obj.get("last_offset")

async def agent_chat(session_id: str, text: str) -> str:
    url = f"{AGENT_API_URL.rstrip('/')}{CHAT_PATH}"
    payload = {"text": text, "session_id": session_id}
    print(f"[bridge] → agent-api {url} payload={payload}")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(url, json=payload)
        print(f"[bridge] agent-api status={r.status_code}")
        r.raise_for_status()
        data = r.json()
    print(f"[bridge] agent-api response={data}")
    if isinstance(data, dict):
        v = data.get("output")
        if isinstance(v, str) and v.strip(): return v
    return str(data)

async def agent_mem(session_id: str) -> str:
    url = f"{AGENT_API_URL.rstrip('/')}{MEM_PATH}"
    print(f"[bridge] → agent-api {url} session_id={session_id}")
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            r = await c.get(url, params={"session_id": session_id})
            print(f"[bridge] agent-api /mem status={r.status_code}")
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        except Exception as e:
            print(f"[bridge] /mem error: {e}")
            return "Memory endpoint not available."

async def agent_clear(session_id: str) -> str:
    url = f"{AGENT_API_URL.rstrip('/')}{CLEAR_PATH}"
    print(f"[bridge] → agent-api {url} session_id={session_id}")
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            r = await c.post(url, json={"session_id": session_id})
            print(f"[bridge] agent-api /clear status={r.status_code}")
            r.raise_for_status()
            return "Memory cleared."
        except Exception as e:
            print(f"[bridge] /clear error: {e}")
            return "Clear endpoint not available."

def is_cmd(text: str) -> bool:
    return text.startswith("/")

def cmd_parts(text: str) -> tuple[str, str]:
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    return cmd, rest

async def handle_command(client: Client, chat_id: str, session_id: str, text: str):
    cmd, rest = cmd_parts(text)
    print(f"[bridge] handling command {cmd} from chat_id={chat_id}")
    if cmd in ("/start", "/resume"):
        if session_id in STOPPED: STOPPED.discard(session_id)
        msg = "Hi there! What can I help you with?\n Use /help to see commands."
        await client.call_tool("send_message", {"chat_id": chat_id, "text": msg})
        return
    if cmd in ("/stop",):
        STOPPED.add(session_id)
        await client.call_tool("send_message", {"chat_id": chat_id, "text": "Paused. Use /resume to continue."})
        return
    if cmd in ("/help",):
        msg = (
            "/start – start or resume\n"
            "/stop – pause responses\n"
            "/mem – show conversation memory\n"
            "/clear – clear memory\n"
            "/help – this help"
        )
        await client.call_tool("send_message", {"chat_id": chat_id, "text": msg})
        return
    if cmd in ("/mem",):
        m = await agent_mem(session_id)
        if not m: m = "No memory."
        if len(m) > 3900: m = m[:3900] + "…"
        await client.call_tool("send_message", {"chat_id": chat_id, "text": str(m)})
        return
    if cmd in ("/clear", "/clear_mem", "/clear_memory"):
        m = await agent_clear(session_id)
        await client.call_tool("send_message", {"chat_id": chat_id, "text": m})
        return
    await client.call_tool("send_message", {"chat_id": chat_id, "text": f"Unknown command: {cmd}. Use /help."})

async def handle_update(client: Client, u: dict):
    chat_id = str(u["chat_id"])
    text = (u["text"] or "").strip()
    print(f"[bridge] update from chat_id={chat_id} text={text}")
    if not text: return
    session_id = f"tg:{chat_id}"
    if is_cmd(text):
        await handle_command(client, chat_id, session_id, text)
        return
    if session_id in STOPPED:
        print(f"[bridge] session {session_id} is stopped, ignoring input")
        await client.call_tool("send_message", {"chat_id": chat_id, "text": "Paused. Use /resume to continue."})
        return
    reply = await agent_chat(session_id, text)
    print(f"[bridge] reply={reply}")
    await client.call_tool("send_message", {"chat_id": chat_id, "text": reply})
    print(f"[bridge] sent message to chat_id={chat_id}")

async def main():
    print(f"[bridge] starting bridge MCP_URL={MCP_URL} AGENT_API_URL={AGENT_API_URL}")
    async with Client(StreamableHttpTransport(url=MCP_URL)) as client:
        await client.ping()
        print("[bridge] connected to MCP server")
        tools = await client.list_tools()
        names = [getattr(t, "name", str(t)) for t in getattr(tools, "tools", tools)]
        print(f"[bridge] available tools={names}")
        assert "get_updates" in names and "send_message" in names
        offset = None
        while True:
            try:
                print(f"[bridge] polling get_updates offset={offset}")
                ups = unwrap(await client.call_tool("get_updates", {"offset": offset, "timeout": TIMEOUT}))
                updates, next_offset = extract_updates(ups)
                print(f"[bridge] got {len(updates)} updates, next_offset={next_offset}")
                for u in updates:
                    try:
                        await handle_update(client, u)
                    except Exception as e:
                        print(f"[bridge] error handling update: {e}")
                        traceback.print_exc()
                if next_offset is not None:
                    offset = next_offset
            except Exception as e:
                print(f"[bridge] main loop error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())