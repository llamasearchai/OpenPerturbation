"""
FastAPI Middleware Configuration

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import time
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Runtime availability flags
FASTAPI_AVAILABLE = False
CORS_AVAILABLE = False

# Import FastAPI components with proper error handling
try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    FASTAPI_AVAILABLE = True
    CORS_AVAILABLE = True
except ImportError:
    logging.warning("FastAPI middleware not available")
    FASTAPI_AVAILABLE = False
    CORS_AVAILABLE = False
    FastAPI = Any
    Request = Any
    CORSMiddleware = Any
    GZipMiddleware = Any

def setup_middleware(app: Any) -> None:
    """
    Configures middleware for the FastAPI application
    
    Args:
        app: FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping middleware setup")
        return
    
    # CORS Middleware
    if CORS_AVAILABLE:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # GZip Middleware
    if CORS_AVAILABLE:  # GZipMiddleware comes with FastAPI
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Any, call_next: Any) -> Any:
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.2f}ms "
            f"status={response.status_code}"
        )
        return response
