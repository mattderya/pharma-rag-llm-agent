import streamlit as st
import os
from agent.rag_pipeline import PharmaRAGPipeline
from agent.langgraph_workflow import run_pharma_query

st.set_page_config(
    page_title="Pharma RAG LLM Agent",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Pharma RAG LLM Agent")
st.caption("HIPAA-Compliant Pharmaceutical AI Assistant | Built by [Matt Derya](https://mattderya.com)")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set!")
    else:
        st.warning("Enter your OpenAI API key to enable full responses.")

    st.divider()
    st.markdown("### 🔬 Example Queries")
    examples = [
        "What are the drug-drug interactions for Warfarin?",
        "Show me Phase II Oncology trial results for Q3",
        "What is the ADME profile of MRD-112?",
        "Which adverse events were most frequent in the Immunology program?",
        "Compare pharmacokinetics of candidates in pipeline",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("LangChain · LangGraph · RAG · FAISS · OpenAI GPT-4")

st.markdown("""
> **Demo Notice:** This app uses mock pharmaceutical data to demonstrate the architecture.
> The production system at Mentor R&D queries proprietary HIPAA-protected clinical databases.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about drug interactions, trial results, ADME profiles...",
                      key="query_input" if "query_input" not in st.session_state else None)

if "query_input" in st.session_state and st.session_state["query_input"]:
    query = st.session_state.pop("query_input")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Running RAG pipeline..."):
            try:
                response = run_pharma_query(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"⚠️ **Demo mode** — API key required for full LLM responses.\n\n*Error: {e}*"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Productivity Gain", "40%", "vs manual queries")
col2.metric("Data Extraction", "Minutes", "from weeks")
col3.metric("Compliance", "HIPAA", "fully maintained")
