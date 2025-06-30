"""API routes for OpenPerturbation."""

from fastapi import APIRouter

from . import analysis, data, jobs, models

# Create a single router to include all sub-routers
api_router = APIRouter()
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(models.router, prefix="/models", tags=["models"])

__all__ = ["api_router"] 