# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building Python packages (e.g., PyMuPDF, ChromaDB)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install CPU-only PyTorch FIRST to avoid downloading ~2GB of NVIDIA CUDA libraries.
# sentence-transformers depends on torch, and pip defaults to the GPU build.
# This container runs on CPU only, so the CPU wheel (~280MB) is all we need.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies (torch is already satisfied, so pip skips it)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH so absolute imports (like 'graph' or 'api') work correctly
ENV PYTHONPATH=/app

# Expose ports for FastAPI (8000) and Hugging Face / Streamlit (7860)
EXPOSE 8000
EXPOSE 7860

# Make the start script executable
RUN chmod +x start.sh

# Default command runs both the FastAPI backend and Streamlit UI
# This is required for single-container PaaS platforms like Hugging Face Spaces
CMD ["./start.sh"]
