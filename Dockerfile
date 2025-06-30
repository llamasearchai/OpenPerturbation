# OpenPerturbation Platform Docker Image
# AI-Driven Perturbation Biology Analysis Platform
# 
# Author: Nik Jois
# Email: nikjois@llamasearch.ai

# Multi-stage Dockerfile for OpenPerturbation
# Author: Nik Jois <nikjois@llamasearch.ai>

# Build stage for dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .
COPY MANIFEST.in .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    APP_ENV=production \
    PORT=8000

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY static/ ./static/
COPY *.py ./
COPY *.toml ./
COPY *.txt ./
COPY *.md ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Testing stage
FROM builder as testing

WORKDIR /app

# Copy test files and additional test dependencies
COPY tests/ ./tests/
COPY pytest.ini .
COPY run_tests.py .

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio pytest-benchmark httpx

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY *.py ./
COPY *.toml ./

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=xml"]

# Development stage
FROM testing as development

# Install development dependencies
RUN pip install --no-cache-dir \
    black \
    isort \
    flake8 \
    mypy \
    pre-commit \
    jupyter \
    ipython

# Install the package in development mode
RUN pip install -e .

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 