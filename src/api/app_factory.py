"""
FastAPI Application Factory

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from typing import Optional, Any, Union

# Import with proper error handling
try:
    from fastapi import FastAPI
    from .middleware import setup_middleware
    from .endpoints import router, FASTAPI_AVAILABLE
    FASTAPI_AVAILABLE_LOCAL = True
except ImportError:
    FASTAPI_AVAILABLE_LOCAL = False
    FastAPI = None
    router = None

def create_app() -> Optional[Any]:
    """
    Creates and configures the FastAPI application instance
    
    Returns:
        FastAPI: Configured application instance, or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE_LOCAL or FastAPI is None:
        return None
        
    app = FastAPI(
        title="OpenPerturbation API",
        description="Comprehensive REST API for perturbation biology analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Import and setup middleware
    try:
        from .middleware import setup_middleware
        setup_middleware(app)
    except ImportError:
        pass  # Middleware is optional
    
    # Include router if available
    if router is not None:
        app.include_router(router, prefix="/api/v1")
    
    return app 