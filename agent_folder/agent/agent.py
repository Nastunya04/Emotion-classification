import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tools import emotion_tool, student_rag_tool

load_dotenv()

SYSTEM = """You are a CLI agent with two tools:
- emotion_classifier: ONLY for explicit emotion/sentiment/feeling/tone classification of a provided text snippet.
- student_rag: ONLY for factual questions about the student. Do not invent facts.

TOOL SELECTION RULES:
1) Call exactly ONE tool per user message unless the user explicitly asks for BOTH.
2) Never call emotion_classifier for biography/factual questions about the student.
3) If the request is unclear, choose the tool that best matches the user's intent.
Be concise.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=0)
    tools = [emotion_tool, student_rag_tool]
    agent = create_tool_calling_agent(llm, tools, PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

_session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

def clear_session_history(session_id: str) -> None:
    _session_store.pop(session_id, None)

def build_agent_with_history() -> RunnableWithMessageHistory:
    base = build_agent()
    return RunnableWithMessageHistory(
        base,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )