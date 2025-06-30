from fastapi import APIRouter, HTTPException
from typing import List, Dict

router = APIRouter()

_DUMMY_DATASETS: Dict[str, Dict] = {
    "demo_dataset": {
        "dataset_id": "demo_dataset",
        "name": "Demo Dataset",
        "description": "Synthetic demo dataset for testing.",
        "num_samples": 1000,
    }
}

@router.get("/", response_model=List[Dict])
async def list_datasets() -> List[Dict]:
    return list(_DUMMY_DATASETS.values())

@router.get("/{dataset_id}", response_model=Dict)
async def get_dataset_info(dataset_id: str):
    if dataset_id not in _DUMMY_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _DUMMY_DATASETS[dataset_id] 