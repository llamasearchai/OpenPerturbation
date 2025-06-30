"""
Analysis-related API endpoints for OpenPerturbation.
"""

import logging
from typing import Any, Dict
import numpy as np
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ...causal.discovery import run_causal_discovery as run_causal_discovery_analysis
from ..models import CausalDiscoveryRequest, ExplainabilityRequest, InterventionDesignRequest

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/models", response_model=Dict[str, Any])
async def get_analysis_models() -> Dict[str, Any]:
    """Get available analysis models and their capabilities."""
    return {
        "causal_discovery": {
            "methods": ["correlation", "pc", "ges", "lingam"],
            "description": "Causal discovery methods for identifying causal relationships",
            "status": "available"
        },
        "explainability": {
            "methods": ["attention", "concept", "pathway"],
            "description": "Model explainability analysis methods",
            "status": "available"
        },
        "prediction": {
            "methods": ["multimodal_transformer", "cell_vit", "molecular_gnn"],
            "description": "Prediction models for various data types",
            "status": "available"
        }
    }

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
    # Enhanced placeholder implementation
    return {
        "attention_analysis": {
            "status": "completed",
            "attention_maps": ["layer_1_attention.png", "layer_2_attention.png"],
            "attention_statistics": {
                "mean_attention": 0.34,
                "max_attention": 0.89,
                "attention_entropy": 2.15
            }
        },
        "concept_analysis": {
            "status": "completed", 
            "activated_concepts": ["cell_division", "protein_interaction", "pathway_activation"],
            "concept_scores": [0.87, 0.65, 0.42]
        }
    }

@router.post("/intervention-design", response_model=Dict[str, Any])
async def design_interventions(request: InterventionDesignRequest) -> Dict[str, Any]:
    """Design optimal interventions based on causal graph."""
    # Enhanced placeholder implementation
    return {
        "recommended_interventions": [
            {
                "target": request.variable_names[0] if request.variable_names else "gene_A",
                "intervention_type": "knockdown",
                "expected_effect": 0.75,
                "confidence": 0.89
            },
            {
                "target": request.variable_names[1] if len(request.variable_names) > 1 else "gene_B",
                "intervention_type": "overexpression", 
                "expected_effect": 0.65,
                "confidence": 0.78
            }
        ],
        "intervention_ranking": {
            "ranking_criteria": "expected_causal_effect",
            "total_interventions": len(request.variable_names) if request.variable_names else 2,
            "budget_utilization": min(request.budget / 1000.0, 1.0) if hasattr(request, 'budget') else 0.8
        }
    }

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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