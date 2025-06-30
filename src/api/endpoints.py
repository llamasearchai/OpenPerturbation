from typing import Dict, Any, List

# Compatibility layer for legacy tests expecting module-level endpoint functions.

async def health_check() -> Dict[str, str]:
    """Legacy async health check used only in tests."""
    return {
        "service": "OpenPerturbation API",
        "status": "healthy",
    }

def list_models() -> List[Dict[str, Any]]:
    """Legacy synchronous model listing for tests."""
    return [{"name": "demo_model", "description": "Stub model"}] 