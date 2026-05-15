import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import ResearchRequest, ResearchResponse
from graph.graph import app as research_app
from storage.vectorstore import delete_pdf_documents, get_vectorstore, reset_vectorstore

app = FastAPI(
    title="Multi-Agent Research Compiler API",
    description="An autonomous AI research engine powered by LangGraph.",
    version="1.0.0"
)

# Enable CORS for the Streamlit UI (or any other client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

# 1. Sync Logic: Remove orphaned chunks for PDFs that no longer exist on disk
# We run this on startup to ensure the vector store matches the local data directory
try:
    vstore = get_vectorstore()
    collection = vstore._collection
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

@app.get("/documents")
async def list_documents():
    """Returns a list of all PDF documents in the data directory."""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document to the data directory."""
    filename = file.filename
    if not filename or not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return {"message": f"Successfully uploaded {filename}", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Deletes a PDF document from the data directory and removes its
    embeddings from the Chroma vector store.
    """
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in data directory.")

    try:
        # Remove embeddings from Chroma first
        chunks_removed = delete_pdf_documents(file_path)
        # Then delete the file from disk
        os.remove(file_path)
        return {
            "message": f"Successfully deleted '{filename}'.",
            "chunks_removed": chunks_removed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/reset")
async def reset_database():
    """
    Wipes the entire vector store and clears the data directory.
    """
    try:
        # 1. Wipe Vector Store
        reset_vectorstore()
        
        # 2. Clear PDF files from disk
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        return {"message": "Database and local files successfully cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/research/stream")
async def conduct_research_stream(request: ResearchRequest):
    """
    Executes the multi-agent research workflow and streams updates back to the client.
    """
    initial_state = {
        "query": request.query,
        "max_iterations": request.max_iterations,
        "sub_tasks": [],
        "web_findings": [],
        "pdf_findings": [],
        "all_findings": [],
        "critic_score": 0.0,
        "critic_feedback": "",
        "critic_sufficient": False,
        "iteration_count": 0,
        "final_report": "",
        "sources": []
    }

    async def event_generator():
        try:
            # Note: We use the synchronous stream wrapper since the graph is likely sync
            # In a fully production async app, you'd use astream()
            for output in research_app.stream(initial_state):
                # Each output is a dict like {"node_name": {state_updates}}
                yield f"{json.dumps(output)}\n"
        except Exception as e:
            yield f"{json.dumps({'error': str(e)})}\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """
    Executes the multi-agent research workflow synchronously.
    """
    try:
        initial_state = {
            "query": request.query,
            "max_iterations": request.max_iterations,
            "sub_tasks": [],
            "web_findings": [],
            "pdf_findings": [],
            "all_findings": [],
            "critic_score": 0.0,
            "critic_feedback": "",
            "critic_sufficient": False,
            "iteration_count": 0,
            "final_report": "",
            "sources": []
        }
        
        final_state = research_app.invoke(initial_state)
        
        return ResearchResponse(
            final_report=final_state.get("final_report", "No report was generated."),
            sources=final_state.get("sources", []),
            iteration_count=final_state.get("iteration_count", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research workflow failed: {str(e)}")
