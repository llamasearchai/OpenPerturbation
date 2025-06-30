from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.post("/validate", response_model=Dict[str, Any])
async def validate_config(config: Dict[str, Any]):
    # Very naive validation: ensure non-empty
    if not config:
        raise HTTPException(status_code=400, detail="Empty configuration")
    # Return structure with warnings list
    warnings = []
    if 'experiment_type' not in config:
        warnings.append("Missing experiment_type field")
    return {
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
        "config": config,
    } 