import fitz  # PyMuPDF
import os
from typing import List, Dict

def load_pdf(file_path: str) -> str:
    """Reads a single PDF using PyMuPDF and returns its text."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"Failed to read PDF {file_path}: {e}")
    return text

def load_all_pdfs(directory: str = "data") -> List[Dict[str, str]]:
    """Loads all PDFs from a given directory."""
    if not os.path.exists(directory):
        # Create directory if it doesn't exist so user knows where to drop PDFs
        os.makedirs(directory, exist_ok=True)
        return []
        
    results = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(directory, filename)
            content = load_pdf(path)
            if content:
                results.append({
                    "title": filename, 
                    "url": path, 
                    "content": content
                })
    return results
