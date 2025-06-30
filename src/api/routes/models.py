"""
Model-related API endpoints for OpenPerturbation.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter

from ...models import MODEL_REGISTRY
from ..models import ModelInfo

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/available", response_model=List[ModelInfo])
def list_available_models() -> List[Dict[str, Any]]:
    """List all available models"""
    if not MODEL_REGISTRY:
        return []
    
    return [
        {
            "name": name,
            "description": getattr(cls, '__doc__', 'No description available.')
        } 
        for name, cls in MODEL_REGISTRY.items()
    ] 