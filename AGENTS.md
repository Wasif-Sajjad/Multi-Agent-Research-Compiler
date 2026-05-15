# Multi-Agent Research Compiler — Project Context

> Drop this file into your project root and reference it at the start of every agent mission.
> Works with Google Antigravity (as a Knowledge Base entry) and Gemini CLI (as GEMINI.md).

---

## What This Project Is

An autonomous AI research system that accepts a natural language question, coordinates a team of specialized AI agents to gather evidence from the web and documents, runs a self-evaluating critic loop, and compiles a polished cited report — all without human intervention.

**Purpose:** AI/ML engineering portfolio project demonstrating LLM orchestration, multi-agent coordination, RAG pipelines, and production-grade system design.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 0.2.28 + LangChain 0.2.16 |
| LLM (free default) | Groq API — Llama 3.1 70B (or Ollama locally) |
| Vector database | Chroma 0.5.3 (local, free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| Web search | DuckDuckGo (free, no key) or Tavily (free tier) |
| API server | FastAPI 0.114.0 + Uvicorn 0.30.6 |
| UI | Streamlit 1.38.0 |
| Observability | LangSmith (free developer tier) |
| Python version | 3.11 or 3.12 |

---

## Project File Structure

```
research-compiler/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py      # Decomposes query → sub-tasks (JSON output)
│   ├── web_search.py        # Tavily / DuckDuckGo → embeds into Chroma
│   ├── pdf_reader.py        # PDF extraction via PyMuPDF → embeds into Chroma
│   ├── critic.py            # Scores findings quality 0–1, returns feedback
│   └── synthesis.py         # Retrieves top-15 chunks, writes cited report
├── graph/
│   ├── __init__.py
│   ├── state.py             # ResearchState TypedDict (source of truth)
│   ├── graph.py             # LangGraph wiring — all nodes + edges
│   └── router.py            # dispatch_agents(), quality_router(), aggregator_node()
├── storage/
│   ├── __init__.py
│   ├── vectorstore.py       # Chroma singleton — shared across all agents
│   └── embeddings.py        # HuggingFaceEmbeddings loader
├── tools/
│   ├── __init__.py
│   ├── search.py            # Unified search: SEARCH_PROVIDER env var switches DDG/Tavily
│   └── pdf_loader.py        # PDF ingestion helpers
├── api/
│   ├── main.py              # FastAPI app, /research POST endpoint
│   └── schemas.py           # ResearchRequest, ResearchResponse Pydantic models
├── ui/
│   └── app.py               # Streamlit interface with graph.stream() live updates
├── eval/
│   ├── benchmark.py         # RAGAS evaluation runner
│   └── test_queries.json    # 20 test research questions
├── tests/
│   ├── test_agents.py
│   ├── test_graph.py
│   └── test_retrieval.py
├── .env                     # API keys — NEVER commit
├── .env.example             # Template for collaborators
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Shared State — The Core Data Model

Every agent reads from and writes to this single TypedDict. **Do not add direct agent-to-agent calls — all communication goes through state.**

```python
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
```

> **Critical:** `Annotated[List[dict], operator.add]` is mandatory on `web_findings` and `pdf_findings`. Without it, parallel branches use last-write-wins and silently discard results.

---

## Agent Roles

| Agent | File | Input from state | Output to state |
|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | `query`, `critic_feedback` | `sub_tasks` |
| Web Search | `agents/web_search.py` | `sub_tasks` (web type) | `web_findings` |
| PDF Reader | `agents/pdf_reader.py` | `sub_tasks` (pdf type) | `pdf_findings` |
| Aggregator | `graph/router.py` | `web_findings`, `pdf_findings` | `all_findings`, `iteration_count` |
| Critic | `agents/critic.py` | `query`, `all_findings` | `critic_score`, `critic_feedback`, `critic_sufficient` |
| Synthesis | `agents/synthesis.py` | `query`, vector store | `final_report`, `sources` |

---

## LangGraph Graph Wiring

```
START
  → orchestrator_node
  → [conditional] dispatch_agents()
      → web_search_node  ─┐
      → pdf_reader_node  ─┤  (parallel, via Send API)
                          ↓
                    aggregator_node
                          ↓
                      critic_node
                          ↓
              [conditional] quality_router()
                ├── score >= 0.75  → synthesis_node → END
                ├── max_iterations → synthesis_node → END
                └── score < 0.75   → orchestrator_node (loop)
```

### Routing functions

```python
# graph/router.py

from langgraph.types import Send

def dispatch_agents(state: ResearchState) -> list:
    """Dynamic parallel branching based on sub-task types."""
    sends = []
    web_tasks = [t for t in state['sub_tasks'] if t['source_type'] == 'web']
    pdf_tasks = [t for t in state['sub_tasks'] if t['source_type'] == 'pdf']
    if web_tasks:
        sends.append(Send('web_search_node', {**state, 'sub_tasks': web_tasks}))
    if pdf_tasks:
        sends.append(Send('pdf_reader_node', {**state, 'sub_tasks': pdf_tasks}))
    if not sends:
        sends.append(Send('web_search_node', state))
    return sends

def quality_router(state: ResearchState) -> str:
    """Conditional edge after critic."""
    if state['critic_sufficient']:
        return 'synthesis_node'
    if state['iteration_count'] >= state['max_iterations']:
        return 'synthesis_node'
    return 'orchestrator_node'
```

---

## Environment Variables

```bash
# .env — copy from .env.example and fill in

# === FREE OPTION A: Groq (fast, no cost) ===
GROQ_API_KEY=your_groq_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile

# === FREE OPTION B: Ollama (fully local) ===
# LLM_PROVIDER=ollama
# LLM_MODEL=llama3.1

# Search (free, no key)
SEARCH_PROVIDER=duckduckgo

# Vector DB (free, local)
VECTOR_DB=chroma
CHROMA_PERSIST_DIR=./chroma_db

# Embeddings (free, local, ~90MB download on first run)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Observability (free tier)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=research-compiler
```

---

## Key Coding Conventions

- Every agent function signature: `def agent_name(state: ResearchState) -> dict`
- Return only the keys the agent modifies — LangGraph merges them
- Wrap all external calls (search, LLM) in `try/except` with `tenacity` retries
- Log with `rich` for readable terminal output during development
- LLM calls use structured JSON output (prompt says "return ONLY JSON, no markdown")
- Chroma is a singleton — import `get_vectorstore()` from `storage/vectorstore.py`, do not instantiate inline
- All prompts live as module-level constants (SCREAMING_SNAKE_CASE), not inline strings

---

## Prompts Reference

### Orchestrator prompt
Decomposes query into 3–5 sub-questions. Returns JSON array: `[{question, source_type: "web"|"pdf", priority: 1-3}]`. Receives `critic_feedback` on loop iterations to refine focus.

### Critic prompt
Evaluates findings against the original query on: coverage, recency, diversity, depth. Returns JSON: `{score: 0.0-1.0, sufficient: bool, gaps: [...], feedback: "..."}`. Threshold: `sufficient = score >= 0.75`.

### Synthesis prompt
Receives numbered context chunks `[1] url\ncontent...`. Writes a structured Markdown report with inline citations like `[1]`. Must ground every claim in the retrieved context.

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Report faithfulness (RAGAS) | ≥ 0.85 |
| Context precision | ≥ 0.75 |
| Critic approval rate (first pass) | 60–70% |
| Average iterations before synthesis | 1.3–1.8 |
| Token cost per run (GPT-4o-mini) | < 15,000 tokens |
| End-to-end latency | < 60 seconds |

---

## Build Order (follow this sequence)

1. `graph/state.py` — ResearchState TypedDict
2. `storage/embeddings.py` + `storage/vectorstore.py` — Chroma singleton
3. `tools/search.py` — unified DuckDuckGo/Tavily interface
4. `agents/orchestrator.py` — query decomposition, verify JSON output
5. `graph/router.py` — `dispatch_agents()` + `aggregator_node()` + `quality_router()`
6. `agents/web_search.py` — search + embed loop
7. `agents/pdf_reader.py` — PyMuPDF extraction + embed loop
8. `agents/critic.py` — quality scoring
9. `agents/synthesis.py` — retrieval + report generation
10. `graph/graph.py` — wire everything with `StateGraph`, test full run
11. `api/main.py` + `api/schemas.py` — FastAPI wrapper
12. `ui/app.py` — Streamlit with `graph.stream()`
13. `eval/benchmark.py` — RAGAS evaluation
14. `tests/` — unit tests per agent
15. `docker-compose.yml` + Hugging Face Spaces deployment

---

## Running the Project

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys

# Run UI
streamlit run ui/app.py

# Run API
uvicorn api.main:app --reload --port 8000

# Run eval
python eval/benchmark.py

# Docker
docker compose up --build
```

---

## What NOT to Do

- Do not call agents directly from other agents — use state
- Do not create a new Chroma instance per agent — use the singleton
- Do not hardcode parallel branches — use `Send()` for dynamic dispatch
- Do not skip the `Annotated` reducer on `web_findings`/`pdf_findings`
- Do not commit `.env` — it is in `.gitignore`
- Do not use Unicode subscript/superscript characters in any output targeting ReportLab PDFs
