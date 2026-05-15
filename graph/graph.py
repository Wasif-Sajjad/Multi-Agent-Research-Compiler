from langgraph.graph import StateGraph, START, END

from graph.state import ResearchState
from graph.router import dispatch_agents, aggregator_node, quality_router

from agents.orchestrator import orchestrator_node
from agents.web_search import web_search_node
from agents.pdf_reader import pdf_reader_node
from agents.critic import critic_node
from agents.synthesis import synthesis_node

def build_graph():
    """
    Constructs and wires the LangGraph workflow.
    """
    # Initialize the graph with our shared state schema
    # We use a type ignore here as some linters are over-strict with TypedDict vs StateT
    workflow = StateGraph(ResearchState)  # type: ignore
    
    # 1. Add all nodes to the graph
    workflow.add_node("orchestrator_node", orchestrator_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("pdf_reader_node", pdf_reader_node)
    workflow.add_node("aggregator_node", aggregator_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("synthesis_node", synthesis_node)
    
    # 2. Entry point: Query starts at the Orchestrator
    workflow.add_edge(START, "orchestrator_node")
    
    # 3. Dynamic Parallel Branching
    # Orchestrator outputs sub_tasks, which are dynamically dispatched using the Send API
    workflow.add_conditional_edges(
        "orchestrator_node", 
        dispatch_agents, 
        ["web_search_node", "pdf_reader_node"]
    )
    
    # 4. Synchronize Parallel Branches
    # Once the agents are done, they both flow into the aggregator node
    workflow.add_edge("web_search_node", "aggregator_node")
    workflow.add_edge("pdf_reader_node", "aggregator_node")
    
    # 5. Aggregator flows to Critic for evaluation
    workflow.add_edge("aggregator_node", "critic_node")
    
    # 6. Quality Control Routing
    # Critic routes either to Synthesis (success/cap reached) or back to Orchestrator (loop)
    workflow.add_conditional_edges(
        "critic_node",
        quality_router,
        {
            "synthesis_node": "synthesis_node",
            "orchestrator_node": "orchestrator_node"
        }
    )
    
    # 7. Synthesis marks the end of the graph
    workflow.add_edge("synthesis_node", END)
    
    # Compile the workflow
    return workflow.compile()

# Expose a compiled instance to be imported by the UI, API, and tests
app = build_graph()
