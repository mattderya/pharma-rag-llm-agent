# ============================================================
# LANGCHAIN AGENT — Pharma LLM Agent
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from agent.rag_pipeline import PharmaRAGPipeline
import os

# Initialize RAG Pipeline
rag = PharmaRAGPipeline()
retriever = rag.get_retriever(k=3)

# ============================================================
# TOOLS
# ============================================================

@tool
def search_pharma_database(query: str) -> str:
    """
    Search the pharmaceutical database for drug information,
    clinical trial results, adverse events, and market data.
    Use this for any pharma-related questions.
    """
    docs = retriever.invoke(query)
    results = []
    for i, doc in enumerate(docs):
        results.append(f"[Source {i+1}: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def get_drug_interactions(drug_name: str) -> str:
    """
    Get drug-drug interactions for a specific pharmaceutical compound.
    Returns interaction profile and clinical recommendations.
    """
    query = f"drug interactions {drug_name} adverse effects"
    docs = retriever.invoke(query)
    
    relevant = [d for d in docs if drug_name.lower() in d.page_content.lower()]
    if relevant:
        return relevant[0].page_content
    return f"No specific interaction data found for {drug_name}. Please consult clinical database."


@tool
def get_clinical_trial_results(program: str) -> str:
    """
    Retrieve clinical trial results for a specific drug program.
    Programs: Oncology, Immunology, or specific drug candidate names.
    """
    query = f"clinical trial results {program} phase endpoints"
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs[:2]])


@tool
def calculate_dose_adjustment(
    drug: str, 
    patient_weight_kg: float, 
    renal_function_crcl: float
) -> str:
    """
    Calculate dose adjustments based on patient parameters.
    Uses pharmacokinetic principles for renal/hepatic impairment.
    """
    adjustment = "Standard dose"
    notes = []
    
    if renal_function_crcl < 30:
        adjustment = "50% dose reduction recommended"
        notes.append("Severe renal impairment detected")
    elif renal_function_crcl < 60:
        adjustment = "25% dose reduction consider"
        notes.append("Moderate renal impairment")
    
    if patient_weight_kg < 50:
        notes.append("Low body weight — monitor closely")
    
    return f"""
    Drug: {drug}
    Patient Weight: {patient_weight_kg} kg
    CrCl: {renal_function_crcl} mL/min
    Recommendation: {adjustment}
    Notes: {', '.join(notes) if notes else 'No adjustments needed'}
    ⚠️ Always verify with clinical pharmacist
    """


# ============================================================
# AGENT
# ============================================================

SYSTEM_PROMPT = """You are a HIPAA-compliant pharmaceutical AI assistant 
built for Mentor R&D's clinical and data science teams.

You have access to:
- Pharmaceutical drug database (interactions, ADME, PK profiles)
- Clinical trial results (Phase I, II, III data)
- Adverse event reports (Oncology, Immunology programs)
- IQVIA market intelligence data
- Dose calculation tools

Always:
1. Cite your data sources
2. Add clinical disclaimers where appropriate
3. Recommend consulting a clinical pharmacist for dosing decisions
4. Flag any serious drug-drug interactions

You are operating in a HIPAA-compliant environment. 
Do not share or store any patient-identifiable information.
"""

def create_pharma_agent(api_key: str = None):
    """Create and return the pharma LLM agent."""
    
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )
    
    tools = [
        search_pharma_database,
        get_drug_interactions,
        get_clinical_trial_results,
        calculate_dose_adjustment
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
