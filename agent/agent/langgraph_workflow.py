# ============================================================
# LANGGRAPH WORKFLOW — Pharma LLM Agent
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from typing import TypedDict, List, Annotated
import operator
import os

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

# ============================================================
# NODES
# ============================================================

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

def classify_query(state: AgentState) -> AgentState:
    """Classify the type of pharmaceutical query."""
    query = state["query"]
    
    keywords_disclaimer = [
        "dose", "dosing", "prescribe", "patient", 
        "treatment", "administer", "mg", "medication"
    ]
    
    requires_disclaimer = any(kw in query.lower() for kw in keywords_disclaimer)
    
    print(f"🔍 Query classified — Disclaimer needed: {requires_disclaimer}")
    
    return {
        **state,
        "requires_disclaimer": requires_disclaimer,
        "iteration": state.get("iteration", 0) + 1
    }

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
    
    prompt = f"""
    Context from pharmaceutical database:
    {state['retrieved_docs']}
    
    Question: {state['query']}
    
    Provide a comprehensive, accurate response based on the context above.
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

def should_retry(state: AgentState) -> str:
    """Decide whether to retry or end the workflow."""
    if not state.get("response") and state.get("iteration", 0) < 3:
        return "retrieve"
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
    workflow.add_node("generate", generate_response)
    
    # Add edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        should_retry,
        {END: END, "retrieve": "retrieve"}
    )
    
    return workflow.compile()


def run_pharma_query(query: str) -> str:
    """Run a pharmaceutical query through the LangGraph workflow."""
    
    app = build_pharma_workflow()
    
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        retrieved_docs="",
        response="",
        requires_disclaimer=False,
        iteration=0
    )
    
    result = app.invoke(initial_state)
    return result["response"]


if __name__ == "__main__":
    # Demo queries
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
