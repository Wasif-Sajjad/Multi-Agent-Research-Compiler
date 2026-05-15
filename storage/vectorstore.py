import os
from langchain_community.vectorstores import Chroma
from storage.embeddings import get_embeddings

# Singleton instance
_VECTORSTORE = None

def get_vectorstore() -> Chroma:
    """
    Returns a singleton instance of the Chroma vector store.
    Ensures that only one connection to the local database is initialized and shared across all agents.
    """
    global _VECTORSTORE
    
    if _VECTORSTORE is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        embeddings = get_embeddings()
        
        _VECTORSTORE = Chroma(
            collection_name="research_compiler",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
    return _VECTORSTORE


def clear_web_findings() -> int:
    """
    Removes all transient web search results from the vector store.
    Call this at the start of a new research run to avoid 'context mixing' 
    from previous unrelated searches.
    """
    vectorstore = get_vectorstore()
    try:
        collection = vectorstore._collection
        results = collection.get(where={"type": "web"})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"Cleared {len(ids_to_delete)} web findings from Chroma.")
        return len(ids_to_delete)
    except Exception as e:
        print(f"Error clearing web findings: {e}")
        return 0


def reset_vectorstore() -> bool:
    """
    Wipes the entire vector store collection. Use with caution.
    """
    vectorstore = get_vectorstore()
    try:
        # Chroma's delete_collection is the safest way to wipe everything
        vectorstore.delete_collection()
        # Reset the singleton so it re-initializes on next call
        global _VECTORSTORE
        _VECTORSTORE = None
        print("Vector store successfully reset.")
        return True
    except Exception as e:
        print(f"Error resetting vector store: {e}")
        return False


def delete_pdf_documents(source_path: str) -> int:
    """
    Removes all document chunks from the Chroma vector store that were
    ingested from a specific PDF file.
    Handles path normalization to ensure consistent matching.
    """
    vectorstore = get_vectorstore()

    # Normalize path (remove leading/trailing spaces, fix slashes)
    norm_path = source_path.replace("\\", "/").strip()

    try:
        collection = vectorstore._collection
        
        # We search for both raw and normalized paths just in case
        results = collection.get(where={"source": source_path})
        ids_to_delete = results.get("ids", [])
        
        if norm_path != source_path:
            norm_results = collection.get(where={"source": norm_path})
            ids_to_delete.extend(norm_results.get("ids", []))

        # Deduplicate IDs
        ids_to_delete = list(set(ids_to_delete))

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks for '{source_path}' from Chroma DB.")
        else:
            print(f"No chunks found in Chroma DB for source '{source_path}'.")

        return len(ids_to_delete)

    except Exception as e:
        print(f"Error deleting PDF documents from vectorstore: {e}")
        return 0
