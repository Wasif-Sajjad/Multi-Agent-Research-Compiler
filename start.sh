#!/bin/bash
set -e

# Start the FastAPI backend in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit UI in the foreground
# Hugging Face Spaces expects the main app to run on port 7860
streamlit run ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true