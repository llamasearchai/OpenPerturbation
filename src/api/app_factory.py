"""
FastAPI Application Factory

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import route modules
from .routes import (
    analysis,
    configuration,
    data,
    datasets,
    experiments,
    jobs,
    models,
    system
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting OpenPerturbation API...")
    yield
    logger.info("Shutting down OpenPerturbation API...")

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="OpenPerturbation API",
        description="AI-Driven Perturbation Biology Analysis Platform",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "OpenPerturbation API", "version": "1.0.0"}
    
    @app.get("/health")
    async def health():
        return {
            "service": "OpenPerturbation API",
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        }
    
    # Include route modules
    app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
    app.include_router(configuration.router, prefix="/configuration", tags=["configuration"])
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
    app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
    app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
    app.include_router(models.router, prefix="/models", tags=["models"])
    app.include_router(system.router, prefix="/system", tags=["system"])
    
    return app 