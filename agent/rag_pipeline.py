# ============================================================
# RAG PIPELINE — Pharma LLM Agent
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
import os

class PharmaRAGPipeline:
    """
    RAG Pipeline for pharmaceutical document retrieval.
    
    In production (Mentor R&D), this pipeline processed:
    - Clinical trial reports
    - Adverse event databases
    - Drug interaction documentation
    - FDA regulatory submissions
    
    This demo uses mock pharmaceutical data with identical architecture.
    """
    
    def __init__(self, data_dir: str = "data/mock_pharma_data"):
        self.data_dir = data_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def load_documents(self) -> list:
        """Load pharmaceutical documents from data directory."""
        documents = []
        
        # Load mock pharma documents
        mock_docs = [
            Document(
                page_content="""
                Drug: Warfarin | Class: Anticoagulant
                Pharmacokinetics: High protein binding (99%), CYP2C9 metabolism
                Drug-Drug Interactions: NSAIDs increase bleeding risk, 
                Rifampin decreases efficacy, Amiodarone increases INR
                Adverse Events: Bleeding, bruising, hair loss
                ADME: Oral bioavailability 100%, Half-life 20-60 hours
                """,
                metadata={"source": "drug_database", "drug": "Warfarin", "category": "anticoagulant"}
            ),
            Document(
                page_content="""
                Phase II Clinical Trial — Oncology Program Q3 2023
                Drug Candidate: MRD-447 | Indication: Non-Small Cell Lung Cancer
                Enrolled: 124 patients | Duration: 6 months
                Primary Endpoint: Overall Response Rate (ORR)
                Results: ORR 42%, Disease Control Rate 78%
                Adverse Events Grade 3+: Fatigue (12%), Nausea (8%), Neutropenia (6%)
                Status: Proceeding to Phase III
                """,
                metadata={"source": "clinical_trials", "phase": "II", "program": "Oncology"}
            ),
            Document(
                page_content="""
                Immunology Program — Adverse Event Report Q3 2023
                Drug: IMM-221 | Indication: Rheumatoid Arthritis
                Total AE Reports: 847 | Serious AEs: 23
                Most Frequent AEs: Injection site reaction (34%), 
                Upper respiratory infection (18%), Headache (12%)
                Drug-Drug Interactions flagged: Methotrexate combination - monitor LFTs
                ADME Profile: SC bioavailability 65%, T1/2 14 days
                """,
                metadata={"source": "adverse_events", "program": "Immunology", "quarter": "Q3"}
            ),
            Document(
                page_content="""
                IQVIA Market Analysis — Oncology Segment Q3 2023
                Total Market Size: $45.2B | YoY Growth: 12.3%
                Top Products by Revenue: Keytruda ($6.1B), Opdivo ($2.1B)
                Pipeline: 847 oncology drugs in development globally
                Key Trends: Combination therapies, Biomarker-driven selection
                Competitive Landscape: 23 new approvals YTD
                """,
                metadata={"source": "iqvia_market", "segment": "Oncology", "quarter": "Q3"}
            ),
            Document(
                page_content="""
                Pharmacokinetics Summary — Drug Candidate MRD-112
                ADME Profile:
                - Absorption: Oral bioavailability 78%, Tmax 2.3 hours
                - Distribution: Vd 45 L/kg, Protein binding 87%
                - Metabolism: CYP3A4 primary, CYP2D6 minor
                - Elimination: Renal 60%, Fecal 35%, T1/2 18 hours
                Drug Interactions: Strong CYP3A4 inhibitors increase AUC 3.2x
                Dose Adjustment: Required in severe renal impairment (CrCl <30)
                """,
                metadata={"source": "pharmacokinetics", "drug": "MRD-112"}
            ),
        ]
        
        documents.extend(mock_docs)
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        print(f"✅ Loaded {len(documents)} documents → {len(splits)} chunks")
        return splits
    
    def build_vector_store(self) -> FAISS:
        """Build FAISS vector store from documents."""
        print("Building vector store...")
        documents = self.load_documents()
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print("✅ Vector store ready")
        return self.vector_store
    
    def get_retriever(self, k: int = 3):
        """Get retriever from vector store."""
        if not self.vector_store:
            self.build_vector_store()
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
