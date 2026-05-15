import os
from typing import Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph.state import ResearchState
from tools.pdf_loader import load_all_pdfs
from storage.vectorstore import get_vectorstore, delete_pdf_documents

def check_relevance(question: str, content: str) -> bool:
    """
    Uses the LLM to strictly determine if a retrieved chunk is relevant to the question.
    """
    from agents.orchestrator import get_llm
    from langchain_core.messages import SystemMessage
    
    llm = get_llm()
    prompt = f"""You are a strict relevance checker.
Evaluate if the following document chunk contains ANY information relevant to answering the question.
If it is relevant, output exactly "YES". If it is completely irrelevant, output exactly "NO".
Do not output anything else.

Question: {question}

Document Chunk:
{content}"""
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return "YES" in str(response.content).upper()
    except Exception as e:
        print(f"Relevance check failed: {e}")
        return False

def pdf_reader_node(state: ResearchState) -> Dict[str, Any]:
    """
    Takes pdf sub-tasks, extracts text from local PDFs, embeds the text into Chroma DB,
    and returns findings to be appended to the state.
    Includes logic to sync the vector store with the local 'data' directory.
    """
    sub_tasks = state.get("sub_tasks", [])
    vectorstore = get_vectorstore()
    
    # 1. Sync Logic: Remove orphaned chunks for PDFs that no longer exist on disk
    try:
        collection = vectorstore._collection
        all_pdf_metadata = collection.get(where={"type": "pdf"}, include=["metadatas"])
        if all_pdf_metadata and all_pdf_metadata["metadatas"]:
            # Ensure metadata is present and source is a string
            metadatas = all_pdf_metadata["metadatas"]
            unique_sources = list(set([str(m["source"]) for m in metadatas if m and "source" in m]))
            for src in unique_sources:
                if src and not os.path.exists(src):
                    print(f"Sync: Removing orphaned chunks for deleted file: {src}")
                    delete_pdf_documents(src)
    except Exception as e:
        print(f"Sync Error: Failed to cleanup orphaned PDF chunks: {e}")

    # We only process if there is a pdf task explicitly requested
    has_pdf_task = any(t.get("source_type") == "pdf" for t in sub_tasks)
    if not has_pdf_task:
        return {"pdf_findings": []}
        
    # Read all PDFs in the 'data' directory
    pdf_docs = load_all_pdfs("data")
    
    if not pdf_docs:
        print("No PDFs found in the 'data' directory to process.")
        return {"pdf_findings": []}
        
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    findings = []
    docs_to_add = []
    collection = vectorstore._collection

    for pdf in pdf_docs:
        source_url = pdf["url"]
        
        # Check if this file is already ingested to prevent duplicates
        # We check by searching for a single chunk with this source
        existing = collection.get(where={"source": source_url}, limit=1)
        if existing and existing["ids"]:
            print(f"Skipping ingestion: {pdf['title']} is already in the vector store.")
            continue

        # Split and prepare for ingestion
        chunks = splitter.split_text(pdf["content"])
        for i, chunk in enumerate(chunks):
            docs_to_add.append(Document(
                page_content=chunk,
                metadata={
                    "source": source_url, 
                    "title": pdf["title"], 
                    "type": "pdf", 
                    "chunk": i
                }
            ))
            
    # Add new documents to Chroma
    if docs_to_add:
        try:
            vectorstore.add_documents(docs_to_add)
            print(f"Successfully embedded {len(docs_to_add)} new PDF chunks into Chroma DB.")
        except Exception as e:
            print(f"Error adding PDFs to vectorstore: {e}")
            
    # Search for relevant chunks for each PDF task
    for task in sub_tasks:
        if task.get("source_type") == "pdf":
            question = task.get("question", "")
            if not question:
                continue
            
            try:
                # Retrieve relevant chunks for this specific sub-task
                results = vectorstore.similarity_search_with_score(
                    question, 
                    k=3, 
                    filter={"type": "pdf"}
                )
                
                for doc, score in results:
                    # Chroma defaults to L2 distance. Lower is better.
                    # A threshold of 1.2 is a reasonable cutoff for relevance.
                    if score < 1.2:
                        # Ask LLM if it's genuinely relevant
                        is_relevant = check_relevance(question, doc.page_content)
                        if is_relevant:
                            findings.append({
                                "title": doc.metadata.get("title", "Unknown"),
                                "url": doc.metadata.get("source", "Unknown"),
                                "content": doc.page_content,
                                "source_type": "pdf"
                            })
                        else:
                            print(f"Discarding irrelevant PDF chunk based on LLM check for: {question}")
                    else:
                        print(f"Discarding irrelevant PDF chunk (score {score:.2f})")
            except Exception as e:
                print(f"Error searching PDF chunks for '{question}': {e}")
                
    return {"pdf_findings": findings}
