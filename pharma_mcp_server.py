# ============================================================
# PHARMA MCP SERVER — Production-Style Tools
# Author: Matt Derya | Data Scientist | mattderya.com
# ============================================================

from mcp.server.fastmcp import FastMCP
from typing import List, Dict

mcp = FastMCP("pharma-server")


# ============================================================
# MOCK PHARMA DATABASE
# (In production: replaced by internal DBs, FDA API, IQVIA, etc.)
# ============================================================

DRUG_INTERACTIONS_DB = {
    "warfarin": [
        {"drug": "NSAIDs", "severity": "High", "effect": "Increased bleeding risk"},
        {"drug": "Rifampin", "severity": "Moderate", "effect": "Decreased efficacy via CYP2C9 induction"},
        {"drug": "Amiodarone", "severity": "High", "effect": "Increased INR, bleeding risk"},
    ],
    "amiodarone": [
        {"drug": "Warfarin", "severity": "High", "effect": "Potentiates anticoagulation"},
        {"drug": "Digoxin", "severity": "Moderate", "effect": "Increased digoxin levels"},
    ],
    "mrd-112": [
        {"drug": "Ketoconazole", "severity": "High", "effect": "3.2-fold AUC increase (CYP3A4 inhibition)"},
        {"drug": "Clarithromycin", "severity": "High", "effect": "Significant AUC increase"},
    ],
}

ADVERSE_EVENTS_DB = {
    "warfarin": [
        {"event": "Bleeding", "frequency": "Common (>10%)", "severity": "Serious"},
        {"event": "Bruising", "frequency": "Common (>10%)", "severity": "Mild"},
    ],
    "mrd-447": [
        {"event": "Fatigue", "frequency": "12%", "severity": "Grade 3+"},
        {"event": "Nausea", "frequency": "8%", "severity": "Grade 3+"},
        {"event": "Neutropenia", "frequency": "6%", "severity": "Grade 3+"},
    ],
    "mrd-112": [
        {"event": "Headache", "frequency": "15%", "severity": "Mild"},
        {"event": "GI upset", "frequency": "9%", "severity": "Mild"},
    ],
}

CLINICAL_TRIALS_DB = [
    {
        "drug": "MRD-447", "phase": "II", "indication": "NSCLC",
        "enrollment": 124, "ORR": "42%", "DCR": "78%", "status": "Proceeding to Phase III"
    },
    {
        "drug": "MRD-112", "phase": "I", "indication": "Solid Tumors",
        "enrollment": 36, "status": "Dose escalation ongoing"
    },
]

PHARMACOKINETICS_DB = {
    "warfarin": {
        "bioavailability": "~100%", "protein_binding": "99%",
        "metabolism": "CYP2C9 (primary)", "half_life": "20-60 hours",
    },
    "mrd-112": {
        "bioavailability": "78%", "Tmax": "2.3 hours",
        "Vd": "45 L/kg", "protein_binding": "87%",
        "metabolism": "CYP3A4 (primary), CYP2D6 (minor)",
        "half_life": "18 hours", "renal_excretion": "60%",
    },
}


# ============================================================
# MCP TOOLS
# ============================================================

@mcp.tool()
def hello_pharma(name: str) -> str:
    """Say hello to someone in pharma context."""
    return f"Hello {name}, welcome to the pharma MCP server!"


@mcp.tool()
def drug_interaction_check(drug_name: str) -> str:
    """
    Check known drug-drug interactions for a given drug.
    Returns structured interaction data including severity and mechanism.
    """
    drug_key = drug_name.lower().strip()
    interactions = DRUG_INTERACTIONS_DB.get(drug_key)
    
    if not interactions:
        return f"No interaction data found for '{drug_name}' in the database."
    
    result = f"Drug Interactions for {drug_name.title()}:\n"
    for interaction in interactions:
        result += f"- {interaction['drug']} | Severity: {interaction['severity']} | Effect: {interaction['effect']}\n"
    return result


@mcp.tool()
def adverse_event_lookup(drug_name: str) -> str:
    """
    Look up adverse events (side effects) for a drug.
    Returns events with frequency and severity classification.
    """
    drug_key = drug_name.lower().strip()
    events = ADVERSE_EVENTS_DB.get(drug_key)
    
    if not events:
        return f"No adverse event data found for '{drug_name}' in the database."
    
    result = f"Adverse Events for {drug_name.title()}:\n"
    for event in events:
        result += f"- {event['event']} | Frequency: {event['frequency']} | Severity: {event['severity']}\n"
    return result


@mcp.tool()
def clinical_trial_search(phase: str = "", indication: str = "") -> str:
    """
    Search clinical trials by phase (I/II/III) and/or indication.
    Leave parameters empty to get all trials.
    """
    results = CLINICAL_TRIALS_DB
    
    if phase:
        results = [t for t in results if t["phase"].lower() == phase.lower().strip()]
    if indication:
        results = [t for t in results if indication.lower() in t["indication"].lower()]
    
    if not results:
        return f"No trials found for phase='{phase}', indication='{indication}'."
    
    output = f"Clinical Trials (phase={phase or 'any'}, indication={indication or 'any'}):\n"
    for trial in results:
        output += f"- {trial['drug']} | Phase {trial['phase']} | {trial['indication']} | n={trial['enrollment']} | Status: {trial['status']}\n"
    return output


@mcp.tool()
def pharmacokinetics_lookup(drug_name: str) -> str:
    """
    Retrieve ADME profile (Absorption, Distribution, Metabolism, Elimination) for a drug.
    """
    drug_key = drug_name.lower().strip()
    pk = PHARMACOKINETICS_DB.get(drug_key)
    
    if not pk:
        return f"No pharmacokinetics data found for '{drug_name}' in the database."
    
    result = f"Pharmacokinetics for {drug_name.title()}:\n"
    for key, value in pk.items():
        result += f"- {key.replace('_', ' ').title()}: {value}\n"
    return result


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()