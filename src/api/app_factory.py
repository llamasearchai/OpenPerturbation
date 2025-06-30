"""
FastAPI Application Factory

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from fastapi import FastAPI
from .middleware import setup_middleware
from .routes import api_router

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
    
    setup_middleware(app)
    app.include_router(api_router)
    
    return app 