# ============================================================
# MULTI-AGENT WORKFLOW — Pharma LLM Agent (A2A Pattern)
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================
# 
# Architecture:
#   Supervisor (LLM router) → {rag_agent, safety_agent, analyst_agent} → Synthesizer
# 
# A2A Communication: shared state's `agent_messages` list
# Every agent appends its findings; downstream agents read them.
# ============================================================

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Annotated
import operator
import sqlite3
import os

load_dotenv()


# ============================================================
# SHARED STATE
# ============================================================

class MultiAgentState(TypedDict):
    """
    Shared state across all agents.
    
    A2A communication happens via `agent_messages`:
    every agent appends its findings; downstream agents read.
    """
    query: str
    agent_messages: Annotated[List[Dict], operator.add]  # A2A channel
    next_agent: str                                       # supervisor routing decision
    active_agents: List[str]                              # agents already called
    final_response: str


# ============================================================
# LLM (shared across all agents)
# ============================================================

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ============================================================
# EXPERT AGENTS
# ============================================================

def rag_agent(state: MultiAgentState) -> Dict:
    """
    RAG Agent — Specialized in document retrieval.
    Reads pharma database (FAISS vector store) and returns relevant chunks.
    """
    from agent.rag_pipeline import PharmaRAGPipeline
    
    print("🔍 [RAG Agent] Searching pharmaceutical database...")
    
    rag = PharmaRAGPipeline()
    retriever = rag.get_retriever(k=3)
    docs = retriever.invoke(state["query"])
    
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'pharma_db')}] {d.page_content}"
        for d in docs
    ])
    
    message = {
        "from": "rag_agent",
        "type": "findings",
        "summary": f"Retrieved {len(docs)} documents",
        "content": context,
    }
    
    print(f"🔍 [RAG Agent] Found {len(docs)} docs → handing off to supervisor")
    
    return {
        "agent_messages": [message],
        "active_agents": state.get("active_agents", []) + ["rag_agent"],
    }


def safety_agent(state: MultiAgentState) -> Dict:
    """
    Safety Agent — Specialized in clinical risk assessment.
    Reads RAG findings and evaluates safety implications.
    """
    print("⚠️  [Safety Agent] Assessing clinical risk...")
    
    # Read prior findings from A2A channel
    prior_findings = "\n".join([
        f"[{m['from']}]: {m.get('content', m.get('summary', ''))}"
        for m in state.get("agent_messages", [])
    ])
    
    system_prompt = """You are a clinical safety specialist.
    Assess pharmaceutical risk: drug interactions, adverse events, contraindications, 
    dosing warnings. Be concise and structured. Flag any HIGH-SEVERITY concerns."""
    
    user_prompt = f"""Query: {state['query']}

Context from other agents:
{prior_findings}

Provide a clinical safety assessment in 3-5 bullet points."""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    message = {
        "from": "safety_agent",
        "type": "safety_assessment",
        "summary": "Clinical risk evaluated",
        "content": response.content,
    }
    
    print(f"⚠️  [Safety Agent] Assessment complete ({len(response.content)} chars)")
    
    return {
        "agent_messages": [message],
        "active_agents": state.get("active_agents", []) + ["safety_agent"],
    }


def analyst_agent(state: MultiAgentState) -> Dict:
    """
    Analyst Agent — Specialized in quantitative data extraction.
    Pulls numbers, metrics, statistics, trial endpoints from context.
    """
    print("📊 [Analyst Agent] Extracting quantitative insights...")
    
    prior_findings = "\n".join([
        f"[{m['from']}]: {m.get('content', m.get('summary', ''))}"
        for m in state.get("agent_messages", [])
    ])
    
    system_prompt = """You are a pharmaceutical data analyst.
    Extract and structure: clinical trial metrics (ORR, DCR, enrollment), 
    pharmacokinetic parameters (Tmax, half-life, bioavailability), 
    incidence rates, statistical significance. Use tables and numbers. 
    If no numeric data is present, say so clearly."""
    
    user_prompt = f"""Query: {state['query']}

Context from other agents:
{prior_findings}

Extract quantitative insights in a structured format."""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    message = {
        "from": "analyst_agent",
        "type": "quantitative_analysis",
        "summary": "Numeric data extracted",
        "content": response.content,
    }
    
    print(f"📊 [Analyst Agent] Analysis complete ({len(response.content)} chars)")
    
    return {
        "agent_messages": [message],
        "active_agents": state.get("active_agents", []) + ["analyst_agent"],
    }

# ============================================================
# SUPERVISOR (LLM-based autonomous routing)
# ============================================================

def supervisor(state: MultiAgentState) -> Dict:
    """
    Supervisor Agent — Routes work to expert agents using LLM reasoning.
    This is the 'autonomous' part: Claude decides which agent runs next.
    """
    print("🧠 [Supervisor] Analyzing query and routing...")
    
    active = state.get("active_agents", [])
    prior_findings = "\n".join([
        f"- [{m['from']}]: {m.get('summary', '')}"
        for m in state.get("agent_messages", [])
    ]) or "(none yet)"
    
    system_prompt = """You are a multi-agent supervisor for a pharmaceutical AI system.
Your job: route queries to the right specialist agent(s) in the right order.

Available agents:
- rag_agent: retrieves documents from pharma database (call FIRST for context)
- safety_agent: assesses clinical risk (drug interactions, adverse events, dosing safety)
- analyst_agent: extracts quantitative data (trial metrics, pharmacokinetics, statistics)

Rules:
1. ALWAYS call rag_agent first if no agents have run yet.
2. Based on the query, decide which other agents add value.
3. Avoid calling the same agent twice.
4. When enough agents have contributed, respond with FINISH.

Respond with EXACTLY one word: rag_agent, safety_agent, analyst_agent, or FINISH."""
    
    user_prompt = f"""Query: {state['query']}

Agents already called: {active if active else '(none)'}

Findings so far:
{prior_findings}

Which agent should act next? (rag_agent / safety_agent / analyst_agent / FINISH)"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    decision = response.content.strip().split()[0].lower().replace(".", "").replace(",", "")
    
    # Normalize decision
    valid = {"rag_agent", "safety_agent", "analyst_agent", "finish"}
    if decision not in valid:
        print(f"🧠 [Supervisor] Unclear decision '{decision}' → defaulting to FINISH")
        decision = "finish"
    
    # Safety: don't call the same agent twice
    if decision != "finish" and decision in active:
        print(f"🧠 [Supervisor] {decision} already called → FINISH")
        decision = "finish"
    
    print(f"🧠 [Supervisor] Decision: → {decision.upper()}")
    
    return {"next_agent": decision}


def route_from_supervisor(state: MultiAgentState) -> str:
    """Conditional edge: reads supervisor's decision and routes accordingly."""
    decision = state.get("next_agent", "finish")
    if decision == "finish":
        return "synthesizer"
    return decision


# ============================================================
# SYNTHESIZER (final response generation)
# ============================================================

def synthesizer(state: MultiAgentState) -> Dict:
    """
    Synthesizer — Aggregates all agent findings into a coherent final response.
    """
    print("✨ [Synthesizer] Merging agent findings into final response...")
    
    all_findings = "\n\n".join([
        f"### {m['from'].replace('_', ' ').title()}\n{m.get('content', m.get('summary', ''))}"
        for m in state.get("agent_messages", [])
    ])
    
    system_prompt = """You are the final synthesizer in a multi-agent pharmaceutical AI system.
Merge findings from all specialist agents into a single, coherent, well-structured response.

Guidelines:
- Attribute insights to the originating agent where relevant
- Resolve conflicts by prioritizing quantitative data over qualitative claims
- Keep clinical safety warnings prominent
- Use markdown: headers, bullets, bold for key numbers"""
    
    user_prompt = f"""Original query: {state['query']}

Contributions from specialist agents:
{all_findings}

Produce the final response for the user."""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    print(f"✨ [Synthesizer] Final response ready ({len(response.content)} chars)")
    
    return {"final_response": response.content}


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_multi_agent_workflow():
    """
    Build and compile the multi-agent LangGraph workflow.
    
    Flow:
        START → supervisor → [rag / safety / analyst] ↺ → synthesizer → END
                    ↑___________________________________|
                    (supervisor decides loop or finish)
    """
    workflow = StateGraph(MultiAgentState)
    
    # Nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("safety_agent", safety_agent)
    workflow.add_node("analyst_agent", analyst_agent)
    workflow.add_node("synthesizer", synthesizer)
    
    # Entry
    workflow.set_entry_point("supervisor")
    
    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "safety_agent": "safety_agent",
            "analyst_agent": "analyst_agent",
            "synthesizer": "synthesizer",
        }
    )
    
    # Every expert agent loops back to supervisor
    workflow.add_edge("rag_agent", "supervisor")
    workflow.add_edge("safety_agent", "supervisor")
    workflow.add_edge("analyst_agent", "supervisor")
    
    # Synthesizer is the terminal node
    workflow.add_edge("synthesizer", END)
    
    # Checkpointer for persistent state
    conn = sqlite3.connect("multi_agent.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    return workflow.compile(checkpointer=checkpointer)


def run_multi_agent_query(query: str, thread_id: str = "multi-agent-session-1") -> str:
    """Run a pharmaceutical query through the multi-agent system."""
    
    app = build_multi_agent_workflow()
    
    initial_state = MultiAgentState(
        query=query,
        agent_messages=[],
        next_agent="",
        active_agents=[],
        final_response="",
    )
    
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15,  # safety cap for supervisor loops
    }
    
    result = app.invoke(initial_state, config=config)
    return result["final_response"]


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    demo_queries = [
        "What are the drug interactions for Warfarin and their severity?",
        "Analyze the Phase II MRD-447 oncology trial — efficacy and safety.",
        "What is the ADME profile of MRD-112 and its key drug interactions?",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'#'*65}")
        print(f"# DEMO QUERY {i}")
        print(f"# {query}")
        print(f"{'#'*65}\n")
        
        response = run_multi_agent_query(
            query,
            thread_id=f"demo-session-{i}"
        )
        
        print(f"\n{'='*65}")
        print("FINAL RESPONSE:")
        print(f"{'='*65}")
        print(response)
        print(f"\n{'='*65}\n")