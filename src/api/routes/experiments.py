from fastapi import APIRouter, HTTPException
from typing import List, Dict

router = APIRouter()

_EXPERIMENTS: Dict[str, Dict] = {
    "exp_001": {
        "experiment_id": "exp_001",
        "name": "Demo Experiment",
        "status": "completed",
    }
}

@router.get("/", response_model=List[Dict])
async def list_experiments() -> List[Dict]:
    return list(_EXPERIMENTS.values())

@router.get("/{experiment_id}", response_model=Dict)
async def get_experiment_info(experiment_id: str):
    if experiment_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _EXPERIMENTS[experiment_id] 