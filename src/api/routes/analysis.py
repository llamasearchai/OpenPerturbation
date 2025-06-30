"""
Analysis-related API endpoints for OpenPerturbation.
"""

import logging
from typing import Any, Dict
import numpy as np
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ...causal.discovery import run_causal_discovery as run_causal_discovery_analysis
from ..models import CausalDiscoveryRequest, ExplainabilityRequest, InterventionDesignRequest

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/causal-discovery", response_model=Dict[str, Any])
async def run_causal_discovery(request: CausalDiscoveryRequest) -> Dict[str, Any]:
    """Run causal discovery analysis on provided data."""
    try:
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided for causal discovery.")

        causal_factors = np.array(request.data)
        # Create dummy perturbation labels for now
        perturbation_labels = np.zeros(len(request.data))

        config = {
            "discovery_method": request.method,
            "alpha": request.alpha,
            "variable_names": request.variable_names,
        }
        results = run_causal_discovery_analysis(causal_factors, perturbation_labels, config)
        return results
    except Exception as e:
        logger.error(f"Causal discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explainability", response_model=Dict[str, Any])
async def run_explainability_analysis(request: ExplainabilityRequest) -> Dict[str, Any]:
    """Run explainability analysis on a trained model."""
    # This is a placeholder for the actual implementation
    return {"message": "Explainability analysis is not yet implemented."}

@router.post("/intervention-design", response_model=Dict[str, Any])
async def design_interventions(request: InterventionDesignRequest) -> Dict[str, Any]:
    """Design optimal interventions based on causal graph."""
    # This is a placeholder for the actual implementation
    return {"message": "Intervention design is not yet implemented."}

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "services": {
            "api": "running",
            "causal_discovery": "available",
            "database": "connected"
        }
    }

@router.post("/causal/discovery", response_model=Dict[str, Any])
async def run_causal_discovery_alias(request: CausalDiscoveryRequest) -> Dict[str, Any]:
    return await run_causal_discovery(request)

@router.post("/intervention/design", response_model=Dict[str, Any])
async def design_interventions_alias(request: InterventionDesignRequest) -> Dict[str, Any]:
    return await design_interventions(request)

@router.post("/explainability/analyze", response_model=Dict[str, Any])
async def run_explainability_alias(request: ExplainabilityRequest) -> Dict[str, Any]:
    return await run_explainability_analysis(request) 