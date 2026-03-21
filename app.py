# ============================================================
# PHARMA RAG LLM AGENT — Main Application
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================

from agent.rag_pipeline import PharmaRAGPipeline
from agent.langgraph_workflow import run_pharma_query
import os

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║       PHARMA RAG LLM AGENT — Mentor R&D Demo         ║
║   HIPAA-Compliant Pharmaceutical AI Assistant         ║
╚══════════════════════════════════════════════════════╝
    """)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable to run with GPT-4")
        print("Running in demo mode with mock responses...\n")

    # Demo queries
    demo_queries = [
        "What are the drug-drug interactions for Warfarin?",
        "Show me Phase II Oncology trial results for Q3",
        "What is the ADME profile of MRD-112?",
        "Which adverse events were most frequent in the Immunology program?",
    ]

    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"🔬 Query: {query}")
        print(f"{'='*60}")
        
        try:
            response = run_pharma_query(query)
            print(f"🤖 Response:\n{response}")
        except Exception as e:
            print(f"Demo mode — API key required for full response: {e}")

if __name__ == "__main__":
    main()
