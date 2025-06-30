"""
FastAPI Application Factory

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from fastapi import FastAPI
from .middleware import setup_middleware
from .routes import api_router
from datetime import datetime

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application instance
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="OpenPerturbation API",
        description="Comprehensive REST API for perturbation biology analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "OpenPerturbation API",
            "version": "1.0.0",
            "status": "running",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "service": "OpenPerturbation API",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    setup_middleware(app)
    # Mount API v1
    app.include_router(api_router, prefix="/api/v1")
    
    return app 