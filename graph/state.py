"""
State definition for the Multi-Agent Research Compiler.

This module defines the ResearchState TypedDict, which acts as the single source
of truth and shared memory across all agents in the LangGraph network.

CRITICAL: The `web_findings` and `pdf_findings` fields use `Annotated[List[dict], operator.add]`.
This is mandatory because the Web Search and PDF Reader agents run in parallel branches.
Without the `operator.add` reducer, LangGraph would use a "last-write-wins" approach,
silently discarding the results from whichever parallel branch finishes first.
The reducer ensures that findings from both branches are safely appended to the list.
"""

from typing import TypedDict, Annotated, List, Optional
import operator

class ResearchState(TypedDict):
    # Input
    query: str                              # Original user question

    # Orchestrator output
    sub_tasks: List[dict]                   # [{question, source_type, priority}]

    # Agent findings — operator.add means parallel branches APPEND, not overwrite
    web_findings: Annotated[List[dict], operator.add]
    pdf_findings: Annotated[List[dict], operator.add]

    # Aggregated
    all_findings: List[dict]                # Merged after parallel phase

    # Critic output
    critic_score: float                     # 0.0 = poor, 1.0 = excellent
    critic_feedback: str                    # Specific gaps for next iteration
    critic_sufficient: bool                 # True → proceed to synthesis

    # Loop control
    iteration_count: int                    # Incremented each loop
    max_iterations: int                     # Hard cap — default 3

    # Final output
    final_report: Optional[str]             # Markdown report
    sources: List[str]                      # Deduplicated source URLs
