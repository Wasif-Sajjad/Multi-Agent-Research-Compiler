import json
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import SystemMessage

from graph.state import ResearchState
from agents.orchestrator import get_llm

CRITIC_PROMPT = """You are an expert research evaluator. Your task is to review the aggregated findings and evaluate if they sufficiently answer the user's original query.
Evaluate the findings based on four criteria:
1. Coverage: Are all aspects of the original query addressed?
2. Recency: Is the information up-to-date (if applicable)?
3. Diversity: Are multiple sources or perspectives included?
4. Depth: Is the information detailed enough to write a comprehensive report?

Return ONLY a JSON object. Do NOT wrap the JSON in markdown blocks (e.g. no ```json).
The JSON object must have exactly these keys:
- "score": (float) A score between 0.0 and 1.0 (where 1.0 is perfect)
- "sufficient": (boolean) True if the score is >= 0.75, False otherwise
- "gaps": (list of strings) 1-3 specific areas where the research is lacking
- "feedback": (string) Specific instructions for the Orchestrator on what to search for next if sufficient is false

Original Query: {query}
Current Findings Summaries: 
{findings}"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_critic_llm(query: str, findings_text: str) -> str:
    llm = get_llm()
    prompt = CRITIC_PROMPT.format(query=query, findings=findings_text)
    
    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content

def critic_node(state: ResearchState) -> Dict[str, Any]:
    """
    Evaluates the aggregated findings against the original query.
    Returns the critic score, feedback, and sufficient flag.
    """
    query = state.get("query", "")
    all_findings = state.get("all_findings", [])
    
    if not all_findings:
        return {
            "critic_score": 0.0,
            "critic_sufficient": False,
            "critic_feedback": "No findings have been gathered yet. Please conduct initial searches."
        }
        
    # Summarize findings for the LLM to avoid exceeding context window
    # We truncate content to 500 chars per finding
    findings_summaries = []
    for i, f in enumerate(all_findings):
        title = f.get("title", "Unknown")
        # Ensure we're dealing with strings, fallback if None
        content = f.get("content") or ""
        content_preview = content[:500] 
        source = f.get("source_type", "unknown")
        findings_summaries.append(f"[{i+1}] {title} ({source}): {content_preview}...")
        
    findings_text = "\n\n".join(findings_summaries)
    
    try:
        raw_response = _call_critic_llm(query, findings_text)
        
        # Parse JSON robustly
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        result = json.loads(cleaned.strip())
        
        score = float(result.get("score", 0.0))
        
        # Enforce threshold: sufficient = score >= 0.75
        sufficient = bool(score >= 0.75)
            
        feedback = result.get("feedback", "")
        gaps = result.get("gaps", [])
        
        if gaps and not sufficient:
            feedback += f"\nIdentified Gaps: {', '.join(gaps)}"
            
        return {
            "critic_score": score,
            "critic_sufficient": sufficient,
            "critic_feedback": feedback
        }
        
    except Exception as e:
        print(f"Critic failed to evaluate: {e}")
        # Fallback to trigger another research iteration
        return {
            "critic_score": 0.5,
            "critic_sufficient": False,
            "critic_feedback": "Critic evaluation failed. Please refine search and gather more comprehensive context."
        }
