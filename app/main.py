# main.py
import os
import pandas as pd
import asyncio
from typing import Dict, Sequence, Annotated
from pydantic import BaseModel
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# langchain / langgraph imports (aligned with your original stack)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

# -------------------------
# Load env + sanity check
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in environment variables.")
print("OPENAI_API_KEY found.")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="FinMentor - Financial Literacy Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for hackathon/prototype. Lock down in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request model
# -------------------------
class SimulateInput(BaseModel):
    message: str
    session_id: str

# -------------------------
# Load dataset & create documents (cached)
# -------------------------
DATA_CSV = "financial_literacy_scenarios.csv"

def load_documents():
    if hasattr(load_documents, "cached"):
        return load_documents.cached

    print("Loading dataset from", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    # Ensure columns exist
    expected = ["situation","risks","consequences","follow_up_questions","skills_required"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # Normalize and construct Documents for RAG
    docs = []
    for _, row in df.iterrows():
        # join fields to create retrieval content
        page_content = (
            f"Situation: {row['situation']}\n"
            f"Risks: {row['risks']}\n"
            f"Consequences: {row['consequences']}\n"
            f"Follow-up Questions: {row['follow_up_questions']}\n"
            f"Skills: {row['skills_required']}"
        )
        docs.append(Document(page_content=page_content, metadata=row.to_dict()))
    load_documents.cached = docs
    return docs

documents = load_documents()

# -------------------------
# Embeddings, text splitting, FAISS index
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
documents_split = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents_split, embeddings)

# -------------------------
# LLM (supervisor) and RAG chain
# -------------------------
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.0)
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=vectorstore.as_retriever())

# -------------------------
# Tools (worker agents)
# -------------------------

@tool
def diagnose_finance(context: str) -> str:
    """
    Return follow-up diagnostic questions or level estimate based on user input.
    Uses vectorstore similarity to find relevant follow-up questions.
    """
    # similarity search for top matches
    docs = vectorstore.similarity_search(context, k=3)
    follow_up_questions = []
    for d in docs:
        q = d.metadata.get("follow_up_questions", "")
        if isinstance(q, str):
            follow_up_questions.extend([s.strip() for s in q.split(";") if s.strip()])
    if follow_up_questions:
        unique_qs = list(dict.fromkeys(follow_up_questions))  # preserve order, unique
        return "Follow-up questions:\n- " + "\n- ".join(unique_qs)
    else:
        return "I couldn't find tailored follow-ups. Could you tell me more about your situation or milestone?"

@tool
def run_simulation(context: str) -> str:
    """
    Produce a short, interactive simulation step or options for the user to choose from.
    The tool uses the dataset retrieval to ground scenario generation.
    """
    # Use retriever to ground simulation choices
    response = retrieval_chain.invoke(f"Create a short, realistic decision step for the user based on: {context}\n"
                                     "Return 3 simple choices, each one short. Begin with 'Simulation:'")
    return response.content.strip()

@tool
def recommend_action(context: str) -> str:
    """
    Given the user's chosen option and context, return a short, concrete recommendation (1-3 sentences).
    Use only dataset-grounded info where possible.
    """
    prompt = (
        "You are a friendly financial assistant for young people. Based only on the provided dataset knowledge, "
        "give a short and actionable recommendation (max 3 sentences) for the following context. "
        "Do not offer medical/legal advice. Include the key action the user should take.\n\n"
        f"Context: {context}\n\nRecommendation:"
    )
    response = retrieval_chain.invoke(prompt)
    return response.content.strip()

@tool
def explain_reasoning(context: str) -> str:
    """
    Explain in simple terms why a recommendation or simulation outcome happened.
    Grounded in dataset (no speculation).
    """
    prompt = (
        "Explain in simple terms why this recommendation or result makes sense, using only dataset-derived knowledge.\n\n"
        f"Context: {context}\n\nExplanation:"
    )
    response = retrieval_chain.invoke(prompt)
    return response.content.strip()

# Bundle tools
tools = [diagnose_finance, run_simulation, recommend_action, explain_reasoning]

# Bind tools to LLM so LangGraph can inspect calls
llm = llm.bind_tools(tools)

# -------------------------
# Define the state & supervisory agent
# -------------------------
class FinAgentState(Sequence, dict):
    """
    We won't rely on a custom TypedDict here — LangGraph expects messages list in the state.
    """
    pass

def fin_supervisor(state: dict) -> dict:
    """
    Supervisory agent: orchestrates diagnosis -> simulation -> recommendation -> explanation
    The system prompt instructs structure and behavior.
    """
    system_prompt = SystemMessage(content="""
You are FinMentor — a friendly, educational financial mentor for youth.
- Start by asking for the user's name, age, and current milestone (e.g., first paycheck, moving out).
- For each tool result or agent output, clearly label which agent produced it:
  (Diagnostic Agent):, (Simulation Agent):, (Recommendation Agent):, (Explanation Agent):
- Ask diagnostic follow-ups one at a time.
- After giving a recommendation, ask the user if they want an explanation.
- Keep language short, encouraging, and age-appropriate.
- Use dataset knowledge only for factual claims. Avoid speculation.
""")
    # Compose messages for LLM: system prompt + conversation history
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    # print tool calls for debugging
    print("[Supervisor] tool calls:", getattr(response, "tool_calls", None))
    return {"messages": [response]}

# -------------------------
# Should-continue check
# -------------------------
def should_continue(state: dict):
    messages = state["messages"]
    last_msg = messages[-1]
    if not getattr(last_msg, "tool_calls", None):
        return "end"
    else:
        return "continue"

# -------------------------
# LangGraph graph assembly
# -------------------------
graph = StateGraph(dict)  # simple dict-based state
graph.add_node("fin_supervisor", fin_supervisor)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("fin_supervisor")

graph.add_conditional_edges("fin_supervisor", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "fin_supervisor")

agent_app = graph.compile()

# -------------------------
# Multi-session conversation store
# -------------------------
conversation_store: Dict[str, list[BaseMessage]] = {}

# -------------------------
# Agent loop runner
# -------------------------
def run_agent_loop(state: dict, session_id: str) -> dict:
    """
    Repeatedly invoke agent graph until no tool calls are present.
    This mirrors your previous 'run_agent_loop' logic.
    """
    local_messages = state["messages"].copy()
    while True:
        # invoke compiled agent_app with current messages
        state_out = agent_app.invoke({"messages": local_messages})
        last_msg = state_out["messages"][-1]
        local_messages.append(last_msg)
        # if last message has no tool_calls, we stop
        if not getattr(last_msg, "tool_calls", None):
            break
    return {"messages": local_messages}

# -------------------------
# FastAPI endpoint
# -------------------------
@app.post("/simulate")
async def simulate(input_data: SimulateInput):
    user_msg = input_data.message.strip()
    session_id = input_data.session_id.strip()

    # init session
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    # append user message to session history
    local_messages = conversation_store[session_id] + [HumanMessage(content=user_msg)]
    state = {"messages": local_messages}

    # run agent in separate thread (non-blocking main event loop)
    state = await asyncio.to_thread(run_agent_loop, state, session_id)

    # update store
    conversation_store[session_id] = state["messages"]

    # return last message content
    last_msg = state["messages"][-1]
    return {"response": last_msg.content}

# -------------------------
# Basic health endpoint
# -------------------------
@app.get("/")
def root():
    return {"status": "FinMentor running", "sessions": len(conversation_store)}
