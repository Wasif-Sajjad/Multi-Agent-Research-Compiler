import os
import json
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from graph.state import ResearchState

ORCHESTRATOR_PROMPT = """You are an expert research orchestrator. Your task is to decompose the user's query into 3-5 sub-tasks for web search and PDF analysis.
Based on the query and any previous feedback, create focused sub-questions that need to be answered.

Return ONLY a JSON array of objects. Do NOT wrap the JSON in markdown blocks (e.g. no ```json).
Each object must have exactly these keys:
- "question": (string) the specific sub-question to research
- "source_type": (string) either "web" or "pdf"
- "priority": (integer) 1, 2, or 3 (1 being highest)

{pdf_context}

Query: {query}
Critic Feedback: {critic_feedback}"""

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model_name = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, temperature=0.1)
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name, temperature=0.1, format="json")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_llm(query: str, critic_feedback: str) -> str:
    from langchain_core.messages import SystemMessage
    from tools.pdf_loader import load_all_pdfs
    
    # Check if PDFs are available
    pdfs = load_all_pdfs("data")
    if pdfs:
        pdf_names = ", ".join([p["title"] for p in pdfs])
        pdf_context = f"Available local PDF files: {pdf_names}. You MAY use 'pdf' as a source_type to search these files if relevant."
    else:
        pdf_context = "No local PDF files are available. You MUST ONLY use 'web' as the source_type. DO NOT output 'pdf'."

    llm = get_llm()
    prompt = ORCHESTRATOR_PROMPT.format(
        query=query, 
        critic_feedback=critic_feedback if critic_feedback else "None",
        pdf_context=pdf_context
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    return str(response.content)

from storage.vectorstore import clear_web_findings

def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Decomposes the user's query into sub-tasks for web and pdf agents.
    Returns a dictionary updating the 'sub_tasks' field in state.
    """
    query = state.get("query", "")
    critic_feedback = state.get("critic_feedback", "")
    iteration = state.get("iteration_count", 0)
    
    # NEW: Clear transient web results at the start of a new research run
    # This prevents 'context mixing' from previous unrelated queries.
    if iteration == 0:
        clear_web_findings()
    
    try:
        raw_response = _call_llm(query, critic_feedback)
        
        # Parse JSON robustly
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        sub_tasks = json.loads(cleaned.strip())
        
        # Ensure correct list format
        if not isinstance(sub_tasks, list):
            if isinstance(sub_tasks, dict) and "sub_tasks" in sub_tasks:
                sub_tasks = sub_tasks["sub_tasks"]
            else:
                sub_tasks = [sub_tasks]
                
        return {"sub_tasks": sub_tasks}
        
    except Exception as e:
        print(f"Orchestrator failed: {e}")
        # Return fallback sub-task to avoid stalling the pipeline
        return {"sub_tasks": [{"question": query, "source_type": "web", "priority": 1}]}
