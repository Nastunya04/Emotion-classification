import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import emotion_tool
from tools import student_rag_tool

load_dotenv()

SYSTEM = """
You are a CLI agent with two tools:

- emotion_classifier: ONLY for explicit emotion/sentiment/feeling/tone classification of a provided text snippet.
- student_rag: ONLY for factual questions about the student. Do not invent facts.

TOOL SELECTION RULES:
1) Call exactly ONE tool per user message unless the user explicitly asks for BOTH (e.g., 'and also classify the emotion').
2) Never call emotion_classifier for biography/factual questions about the student.
3) If the request is unclear, choose the tool that best matches the user's intent.

Be concise.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

def build_agent():
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL","gpt-4o-mini"), temperature=0)
    tools = [emotion_tool, student_rag_tool]
    agent = create_tool_calling_agent(llm, tools, PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

agent_executor = build_agent()