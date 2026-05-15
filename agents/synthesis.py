import os
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import SystemMessage

from graph.state import ResearchState
from storage.vectorstore import get_vectorstore

# We define a dedicated LLM factory because Synthesis MUST output Markdown,
# whereas orchestrator/critic often force JSON mode (especially in Ollama).
def get_synthesis_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model_name = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        # Slightly higher temperature (0.2) for better narrative flow in the report
        return ChatGroq(model=model_name, temperature=0.2)
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        # Crucial: NO format="json" here. We want raw markdown text.
        return ChatOllama(model=model_name, temperature=0.2)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

SYNTHESIS_PROMPT = """You are an expert research analyst. Your task is to write a comprehensive, polished, and well-structured markdown report answering the user's original query.

You must rely ONLY on the numbered context chunks provided below. 
Every claim or fact you state MUST be grounded in these chunks.
You MUST include inline citations using the exact bracketed numbers (e.g. [1], [3]) corresponding to the chunks you used.

The report should have:
1. A clear, engaging # Title
2. An Executive Summary
3. Well-structured sub-sections (using ## or ###) based on the findings
4. A unified Conclusion

Do not hallucinate any information. If the provided context does not cover part of the query, state that the information is unavailable.

Original Query: {query}

Numbered Context Chunks:
{context}"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_synthesis_llm(query: str, context: str) -> str:
    llm = get_synthesis_llm()
    prompt = SYNTHESIS_PROMPT.format(query=query, context=context)
    
    response = llm.invoke([SystemMessage(content=prompt)])
    return str(response.content)

def synthesis_node(state: ResearchState) -> Dict[str, Any]:
    """
    Retrieves the top-15 highest-relevance chunks from Chroma AND includes 
    all direct findings from the current research session to author 
    the final, fully-cited markdown report.
    Returns the final_report and the list of deduplicated sources.
    """
    query = state.get("query", "")
    all_findings = state.get("all_findings", [])
    vectorstore = get_vectorstore()
    
    try:
        # Retrieve top 15 most relevant chunks from the vector database
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=15)
        # Filter out irrelevant chunks
        docs = [doc for doc, score in docs_and_scores if score < 1.2]
    except Exception as e:
        print(f"Synthesis failed to retrieve documents from Chroma: {e}")
        docs = []
        
    context_parts = []
    sources = []
    seen_urls = set()
    seen_content = set()
    
    chunk_id = 1
    
    # 1. Incorporate Chroma vector DB documents
    for doc in docs:
        url = doc.metadata.get("source", "Unknown Source")
        title = doc.metadata.get("title", "Unknown Title")
        content = doc.page_content
        
        if content not in seen_content:
            seen_content.add(content)
            context_parts.append(f"[{chunk_id}] URL/Source: {url}\nTitle: {title}\nContent:\n{content}")
            chunk_id += 1
            if url not in seen_urls and url != "Unknown Source":
                seen_urls.add(url)
                sources.append(url)

    # 2. Incorporate direct findings from the current research session
    for finding in all_findings:
        url = finding.get("url", "Unknown Source")
        title = finding.get("title", "Unknown Title")
        content = finding.get("content", "")
        
        if content and content not in seen_content:
            seen_content.add(content)
            context_parts.append(f"[{chunk_id}] URL/Source: {url}\nTitle: {title}\nContent:\n{content}")
            chunk_id += 1
            if url not in seen_urls and url != "Unknown Source":
                seen_urls.add(url)
                sources.append(url)
                
    if not context_parts:
        return {
            "final_report": "I could not find enough relevant information to generate a report on this topic.",
            "sources": []
        }
            
    context_text = "\n\n".join(context_parts)
    
    try:
        final_report = _call_synthesis_llm(query, context_text)
    except Exception as e:
        print(f"Synthesis LLM failed: {e}")
        final_report = "Error generating the report due to a language model failure."
        
    return {
        "final_report": final_report,
        "sources": sources
    }
