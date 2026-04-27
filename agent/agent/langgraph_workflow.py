# ============================================================
# LANGGRAPH WORKFLOW — Pharma LLM Agent
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Annotated
import operator
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import asyncio

# MCP imports (updated for clean subprocess lifecycle)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()


# ============================================================
# STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    """State for the pharmaceutical AI agent workflow."""
    messages: Annotated[List, operator.add]
    query: str
    retrieved_docs: str
    response: str
    requires_disclaimer: bool
    iteration: int
    mcp_results: str


# ============================================================
# LLM
# ============================================================

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ============================================================
# NODES
# ============================================================

def classify_query(state: AgentState) -> AgentState:
    """Classify the type of pharmaceutical query."""
    query = state["query"]
    
    keywords_disclaimer = [
        "dose", "dosing", "prescribe", "patient",
        "treatment", "administer", "mg", "medication"
    ]
    
    requires_disclaimer = any(kw in query.lower() for kw in keywords_disclaimer)
    
    print(f"🔍 Query classified — Disclaimer needed: {requires_disclaimer}")
    
    return {**state, "requires_disclaimer": requires_disclaimer}


def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant pharmaceutical context via RAG."""
    from agent.rag_pipeline import PharmaRAGPipeline
    
    rag = PharmaRAGPipeline()
    retriever = rag.get_retriever(k=3)
    
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'pharma_db')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    print(f"📚 Retrieved {len(docs)} relevant documents")
    
    return {**state, "retrieved_docs": context}


# ============================================================
# MCP TOOL HELPERS
# ============================================================

def _extract_text(mcp_result) -> str:
    """Extract plain text from MCP tool result (list of content blocks)."""
    if isinstance(mcp_result, list):
        texts = []
        for block in mcp_result:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            else:
                texts.append(str(block))
        return "\n".join(texts)
    return str(mcp_result)


def _extract_drug_name(query: str) -> str:
    """Extract a known drug name from the query."""
    known_drugs = ["warfarin", "amiodarone", "mrd-112", "mrd-447"]
    for drug in known_drugs:
        if drug in query.lower():
            return drug
    return ""


async def _call_mcp_async(query: str) -> str:
    """
    Async helper: connect to MCP server and invoke the relevant tool.
    Uses explicit async context managers to guarantee subprocess cleanup
    after each query (prevents leak/freeze across multiple runs).
    """
    server_params = StdioServerParameters(
        command="python",
        args=["pharma_mcp_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tool_map = {t.name: t for t in tools}
            
            q = query.lower()
            
            if "interaction" in q:
                tool = tool_map.get("drug_interaction_check")
                drug = _extract_drug_name(q) or "warfarin"
                raw = await tool.ainvoke({"drug_name": drug})
                return _extract_text(raw)
            elif "adverse" in q or "side effect" in q:
                tool = tool_map.get("adverse_event_lookup")
                drug = _extract_drug_name(q) or "warfarin"
                raw = await tool.ainvoke({"drug_name": drug})
                return _extract_text(raw)
            elif "trial" in q or "phase" in q:
                tool = tool_map.get("clinical_trial_search")
                phase = "II" if "phase ii" in q or "phase 2" in q else ""
                indication = "NSCLC" if "nsclc" in q or "oncology" in q else ""
                raw = await tool.ainvoke({"phase": phase, "indication": indication})
                return _extract_text(raw)
            elif "adme" in q or "pharmacokinetic" in q or "metabolism" in q:
                tool = tool_map.get("pharmacokinetics_lookup")
                drug = _extract_drug_name(q) or "mrd-112"
                raw = await tool.ainvoke({"drug_name": drug})
                return _extract_text(raw)
            
            return ""


def call_mcp_tools(state: AgentState) -> AgentState:
    """Invoke MCP tools based on query keywords and enrich state."""
    query = state["query"]
    
    try:
        result = asyncio.run(_call_mcp_async(query))
        if result:
            print(f"🔧 MCP tool invoked — {len(result)} chars returned")
        else:
            print(f"⏭️  No MCP tool matched for this query")
        return {**state, "mcp_results": result}
    except Exception as e:
        print(f"⚠️  MCP tool call failed: {e}")
        return {**state, "mcp_results": ""}


def generate_response(state: AgentState) -> AgentState:
    """Generate response using LLM with retrieved context."""
    
    system_prompt = """You are a HIPAA-compliant pharmaceutical AI assistant 
    for Mentor R&D clinical and data science teams.
    
    Use the provided context to answer questions about:
    - Drug interactions and adverse events
    - Clinical trial results
    - Pharmacokinetics and ADME profiles
    - Market intelligence (IQVIA data)
    
    Always cite your sources and maintain clinical accuracy."""
    
    mcp_context = state.get("mcp_results", "")
    mcp_section = f"\n    Structured data from MCP tools:\n    {mcp_context}\n" if mcp_context else ""
    
    prompt = f"""
    Context from pharmaceutical database (RAG):
    {state['retrieved_docs']}
    {mcp_section}
    Question: {state['query']}
    
    Provide a comprehensive, accurate response using the context above.
    If structured MCP data is available, prioritize it for factual precision (dosing, interactions, trial numbers).
    """
    
    response = llm.invoke([
        HumanMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    answer = response.content
    
    if state.get("requires_disclaimer"):
        answer += "\n\n⚠️ CLINICAL DISCLAIMER: This information is for research purposes only. Always verify dosing decisions with a licensed clinical pharmacist or physician."
    
    print(f"✅ Response generated ({len(answer)} chars)")
    
    return {**state, "response": answer}


def increment_iteration(state: AgentState) -> AgentState:
    """Increment the retry iteration counter."""
    current = state.get("iteration", 0)
    new_iteration = current + 1
    print(f"🔄 Retry iteration: {new_iteration}")
    return {**state, "iteration": new_iteration}


def should_retry(state: AgentState) -> str:
    """Decide whether to retry or end the workflow."""
    if not state.get("response") and state.get("iteration", 0) < 3:
        return "retry"
    return END


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_pharma_workflow():
    """Build and compile the LangGraph workflow."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("call_mcp_tools", call_mcp_tools)
    workflow.add_node("generate", generate_response)
    workflow.add_node("increment_iteration", increment_iteration)

    # Add edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "call_mcp_tools")
    workflow.add_edge("call_mcp_tools", "generate")
    workflow.add_conditional_edges(
        "generate",
        should_retry,
        {END: END, "retry": "increment_iteration"}
    )
    workflow.add_edge("increment_iteration", "retrieve")

    # Checkpointer for persistent conversation state
    conn = sqlite3.connect("pharma_agent.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    return workflow.compile(checkpointer=checkpointer)


def run_pharma_query(query: str) -> str:
    """Run a pharmaceutical query through the LangGraph workflow."""
    
    app = build_pharma_workflow()
    
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        retrieved_docs="",
        response="",
        requires_disclaimer=False,
        iteration=0,
        mcp_results=""
    )
    
    config = {"configurable": {"thread_id": "pharma-session-1"}}
    result = app.invoke(initial_state, config=config)
    return result["response"]


if __name__ == "__main__":
    queries = [
        "What are the drug interactions for Warfarin?",
        "Show me Phase II Oncology trial results",
        "What is the ADME profile of MRD-112?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        response = run_pharma_query(query)
        print(f"Response: {response}")