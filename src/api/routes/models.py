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

@router.get("/", response_model=List[ModelInfo])
def list_models_root() -> List[Dict[str, Any]]:
    return list_available_models() 

@router.get("/{model_id}", response_model=Dict[str, Any])
def get_model_info(model_id: str):
    if model_id not in MODEL_REGISTRY:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Model not found")
    cls = MODEL_REGISTRY[model_id]
    return {
        "model_id": model_id,
        "name": cls.__name__,
        "description": getattr(cls, '__doc__', ''),
        "status": "available",
    } 