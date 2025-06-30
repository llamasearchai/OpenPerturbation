"""
API Module Initialization

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

# Import only the names that should be exported
# Expose the consolidated API router
try:
    from .routes import api_router as router  # noqa: F401
except ImportError:  # Fallback if routes package missing during minimal envs
    router = None
from .middleware import setup_middleware
from .app_factory import create_app

# Only include names that are actually defined
__all__ = [
    "router",
    "setup_middleware",
    "create_app"
]

def get_app():
    """Get the FastAPI app instance."""
    from .server import app
    return app

def get_create_app():
    """Get the create_app function."""
    from .server import create_app
    return create_app

def get_router():
    """Get the API router (if available)."""
    return router

def get_middleware_setup():
    """Get the middleware setup function."""
    from .middleware import setup_middleware
    return setup_middleware
