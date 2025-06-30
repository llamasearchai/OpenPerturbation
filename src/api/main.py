"""
Main application entry point

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from typing import Optional, Any
from .app_factory import create_app

# Create the app instance with proper type handling
app_instance = create_app()

# Only assign to app if we have a valid instance
if app_instance is not None:
    app = app_instance
else:
    # Create a minimal fallback app for when FastAPI is not available
    class FallbackApp:
        def __call__(self, *args, **kwargs):
            return {"error": "FastAPI not available"}
    
    app = FallbackApp()

if __name__ == "__main__":
    try:
        import uvicorn
        if app_instance is not None:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print("FastAPI not available - cannot start server")
    except ImportError:
        print("Uvicorn not available - cannot start server") 