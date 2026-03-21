# 🤖 Pharma RAG LLM Agent
### HIPAA-Compliant Agentic AI for Pharmaceutical Enterprise Data Querying

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-412991.svg)](https://openai.com)
[![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-red.svg)](https://hhs.gov)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

During my work at **Mentor R&D**, I architected and deployed **HIPAA-compliant LLM agents** using **LangChain/LangGraph/RAG** for enterprise pharmaceutical data querying and Agentic AI workflows.

The system enabled non-technical stakeholders to query complex clinical trial databases using natural language — eliminating weeks of manual data extraction and improving team productivity by **40%**.

**This repository demonstrates the core architecture using publicly shareable code.**

---

## ⚠️ Replica Notice

> The original system queried **proprietary clinical trial databases** and **adverse event reports** protected under **HIPAA compliance**. This repo demonstrates the same architecture using mock pharmaceutical data.

| | This Repo | Production System |
|---|---|---|
| **Data** | Mock pharma data | Proprietary clinical databases |
| **LLM** | OpenAI GPT-4 (configurable) | Fine-tuned enterprise LLM |
| **Scale** | Demo | Enterprise deployment on AWS |
| **Auth** | API key | HIPAA-compliant SSO |

---

## 📊 Production Results (Mentor R&D)

- 🚀 **40% productivity improvement** across data science and clinical teams
- ⚡ Manual data extraction time reduced from **weeks → minutes**
- 🔒 Full **HIPAA compliance** maintained throughout
- 🤖 Autonomous **multi-step reasoning** over clinical trial data
- ☁️ Deployed on **AWS (EC2, SageMaker)** with CI/CD pipelines

---

## 🏗️ Architecture
```
User Query
    ↓
LangGraph Orchestrator (Agentic Workflow)
    ↓
┌─────────────────────────────────────┐
│  RAG Pipeline                        │
│  ├── Document Loader                 │
│  ├── Text Splitter                   │
│  ├── Embeddings (OpenAI/HuggingFace) │
│  └── Vector Store (FAISS/Chroma)     │
└─────────────────────────────────────┘
    ↓
LangChain Agent + Tools
    ├── SQL Query Tool
    ├── Document Search Tool
    └── Calculation Tool
    ↓
Structured Response
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **LLM Framework** | LangChain, LangGraph |
| **LLM Models** | OpenAI GPT-4, Claude |
| **RAG** | FAISS, Chroma, HuggingFace Embeddings |
| **Backend** | Python, FastAPI |
| **Cloud** | AWS (EC2, SageMaker) |
| **Compliance** | HIPAA, FDA Regulatory |
| **Domain** | Pharmacokinetics, ADME, Oncology, Immunology |

---

## 📁 Repository Structure
```
pharma-rag-llm-agent/
│
├── agent/
│   ├── rag_pipeline.py        # RAG setup & vector store
│   ├── langchain_agent.py     # LangChain agent & tools
│   └── langgraph_workflow.py  # LangGraph orchestration
│
├── data/
│   └── mock_pharma_data/      # Sample pharmaceutical data
│
├── app.py                     # Main application entry point
├── requirements.txt           # Dependencies
└── README.md
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/mattderya/pharma-rag-llm-agent.git
cd pharma-rag-llm-agent
pip install -r requirements.txt

# Add your OpenAI API key
export OPENAI_API_KEY="your-key-here"

python app.py
```

---

## 💬 Example Queries
```python
# Natural language queries the agent can handle:
"What are the drug-drug interactions for patients on Warfarin?"
"Show me Phase II trial results for Oncology program Q3"
"Compare ADME profiles across our current drug candidates"
"Which adverse events were reported most frequently last month?"
```

---

## 👤 Author

**Matt Derya** | Data Scientist | 15+ years Pharma
- 🌐 [mattderya.com](https://mattderya.com)
- 💼 [linkedin.com/in/mttdryai](https://linkedin.com/in/mttdryai)
- 🐙 [github.com/mattderya](https://github.com/mattderya)
- 📧 mttdryai@gmail.com
