"""
Job-related API endpoints for OpenPerturbation.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..models import AnalysisRequest

router = APIRouter()
logger = logging.getLogger(__name__)

_JOBS: Dict[str, Dict[str, Any]] = {}

@router.post("/start", response_model=Dict[str, Any])
async def start_analysis_job(request: AnalysisRequest):
    """Launches an analysis pipeline."""
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "message": "Analysis started",
        "results": {},
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    return {"job_id": job_id, "status": "queued", "message": "Analysis started successfully"}

@router.get("/{job_id}/status", response_model=Dict[str, Any])
async def get_analysis_status(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _JOBS[job_id]

    # Simulate progress
    if job["status"] == "running":
        job["progress"] += 50
        if job["progress"] >= 100:
            job["progress"] = 100
            job["status"] = "completed"
            job["results"] = {"summary": "Mock analysis results"}
        job["updated_at"] = datetime.utcnow().isoformat()

    return {
        "id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "results": job["results"],
    }

@router.delete("/{job_id}", response_model=Dict[str, Any])
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    _JOBS[job_id]["status"] = "cancelled"
    _JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": f"Job {job_id} has been cancelled"
    } 