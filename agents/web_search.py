from typing import Dict, Any
from langchain_core.documents import Document

from graph.state import ResearchState
from tools.search import search
from storage.vectorstore import get_vectorstore

def web_search_node(state: ResearchState) -> Dict[str, Any]:
    """
    Takes web sub-tasks, performs the searches, embeds the text into the Chroma DB,
    and returns findings to be appended to the state.
    """
    # The router sends only the web sub-tasks here via the Send API
    sub_tasks = state.get("sub_tasks", [])
    
    findings = []
    docs_to_add = []
    
    # Singleton vector store
    vectorstore = get_vectorstore()
    
    for task in sub_tasks:
        question = task.get("question", "")
        if not question:
            continue
            
        try:
            # Execute search with retries via our search tool
            # Adjust max_results as needed, 3 is usually good for broad coverage per query
            results = search(question, max_results=3)
            
            for res in results:
                title = res.get("title", "")
                url = res.get("url", "")
                content = res.get("content", "")
                
                if not content or not url:
                    continue
                    
                # 1. Append to findings (for Critic evaluation)
                findings.append({
                    "title": title,
                    "url": url,
                    "content": content,
                    "source_type": "web"
                })
                
                # 2. Add to vector store buffer (for Synthesis retrieval)
                doc = Document(
                    page_content=content,
                    metadata={"source": url, "title": title, "type": "web"}
                )
                docs_to_add.append(doc)
                
        except Exception as e:
            print(f"Error during web search for '{question}': {e}")
            
    # Batch add all found chunks into the Chroma database
    if docs_to_add:
        try:
            vectorstore.add_documents(docs_to_add)
        except Exception as e:
            print(f"Error adding web documents to vectorstore: {e}")
            
    # Return the findings to trigger the operator.add reducer in ResearchState
    return {"web_findings": findings}
