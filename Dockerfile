# OpenPerturbation Platform Docker Image
# AI-Driven Perturbation Biology Analysis Platform
# 
# Author: Nik Jois
# Email: nikjois@llamasearch.ai

# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r openperturbation && useradd -r -g openperturbation openperturbation

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads/genomics uploads/imaging uploads/chemical \
    outputs test_output demo_output logs temp cache checkpoints

# Set permissions
RUN chown -R openperturbation:openperturbation /app && \
    chmod +x src/api/server.py

# Switch to non-root user
USER openperturbation

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "src.api.server", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx \
    black \
    isort \
    flake8 \
    mypy \
    bandit

# Set development environment
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=DEBUG
ENV DEBUG=true

# Development command
CMD ["python", "-m", "src.api.server", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Set production environment
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO
ENV DEBUG=false

# Production command
CMD ["python", "-m", "src.api.server", "--host", "0.0.0.0", "--port", "8000"]

# Testing stage
FROM base as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx \
    black \
    isort \
    flake8 \
    mypy

# Set testing environment
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=WARNING

# Run tests
CMD ["python", "run_tests.py", "--all", "--coverage"] 