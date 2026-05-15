import os
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

def _duckduckgo_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    from duckduckgo_search import DDGS
    
    formatted = []
    with DDGS() as ddgs:
        # DDGS.text yields dictionaries with 'href', 'title', and 'body'
        results = list(ddgs.text(query, max_results=max_results))
        for r in results:
            formatted.append({
                "url": r.get("href", ""),
                "title": r.get("title", ""),
                "content": r.get("body", "")
            })
    return formatted

def _tavily_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    
    if not isinstance(results, list):
        return []

    formatted = []
    for r in results:
        formatted.append({
            "url": r.get("url", ""),
            "title": r.get("title", "Tavily Search Result"),
            "content": r.get("content", "")
        })
    return formatted

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Unified search interface that routes to the configured search provider.
    Reads SEARCH_PROVIDER from environment (defaults to duckduckgo).
    Returns a list of dictionaries with 'url', 'title', and 'content' keys.
    """
    provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    
    try:
        if provider == "tavily":
            return _tavily_search(query, max_results)
        else:
            return _duckduckgo_search(query, max_results)
    except Exception as e:
        print(f"Search failed for query '{query}' using {provider}: {e}")
        raise  # Re-raise to trigger tenacity retry
