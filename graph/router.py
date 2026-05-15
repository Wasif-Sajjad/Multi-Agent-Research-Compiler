import os
from langgraph.types import Send
from typing import List, Dict, Any
from graph.state import ResearchState

def dispatch_agents(state: ResearchState) -> List[Send]:
    """
    Dynamic parallel branching based on sub-task types.
    Routes 'web' tasks to the web_search_node and 'pdf' tasks to the pdf_reader_node.
    Strictly enforces that pdf tasks are converted to web tasks if no PDF files exist.
    """
    sends = []
    sub_tasks = state.get("sub_tasks", [])
    
    # Check if any PDFs exist
    has_pdfs = False
    if os.path.exists("data"):
        has_pdfs = any(f.lower().endswith('.pdf') for f in os.listdir("data"))
    
    web_tasks = []
    pdf_tasks = []
    
    for t in sub_tasks:
        if t.get("source_type") == "pdf" and has_pdfs:
            pdf_tasks.append(t)
        else:
            # If no PDFs exist, fallback to web search even if orchestrator said 'pdf'
            # Or if it was already 'web'
            web_tasks.append(t)
    
    if web_tasks:
        sends.append(Send("web_search_node", {**state, "sub_tasks": web_tasks}))
    
    if pdf_tasks:
        sends.append(Send("pdf_reader_node", {**state, "sub_tasks": pdf_tasks}))
        
    # Fallback to web search if no valid types are found
    if not sends:
        sends.append(Send("web_search_node", state))
        
    return sends

def aggregator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Collects state after parallel branches.
    Merges web_findings and pdf_findings into all_findings.
    Increments iteration_count.
    """
    web_findings = state.get("web_findings", [])
    pdf_findings = state.get("pdf_findings", [])
    
    # Merge both lists. Because web_findings and pdf_findings use operator.add,
    # they naturally accumulate findings across multiple loop iterations as well.
    all_findings = web_findings + pdf_findings
    
    current_iter = state.get("iteration_count", 0)
    
    return {
        "all_findings": all_findings,
        "iteration_count": current_iter + 1
    }

def quality_router(state: ResearchState) -> str:
    """
    Conditional edge after the critic node.
    Decides whether to synthesize the final report or loop back for more research.
    """
    if state.get("critic_sufficient", False):
        return "synthesis_node"
    
    # Enforce hard cap on iterations
    current_iter = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    
    if current_iter >= max_iter:
        return "synthesis_node"
        
    return "orchestrator_node"
