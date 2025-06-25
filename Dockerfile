# Use official Python 3.9 image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=cpu
ENV FORCE_CUDA=0
ENV CUDA_HOME=""

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    cmake \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install torch and numpy first (required for some packages)
RUN pip install --no-cache-dir torch>=2.0.0 numpy

# Install auto-gptq v0.4.2 from GitHub (CPU-only compatible)
RUN pip install --no-cache-dir git+https://github.com/ModelCloud/GPTQModel.git

# Install AWQ from local clone
COPY llm-awq /tmp/llm-awq
RUN pip install --no-cache-dir /tmp/llm-awq

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies (excluding torch, numpy, auto-gptq)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs model_cache logs

# Expose ports
EXPOSE 8000 6379

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "start.py", "--mode", "prod"] 