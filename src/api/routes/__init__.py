"""API routes for OpenPerturbation."""

from fastapi import APIRouter

from . import analysis, data, jobs, models, datasets, experiments, configuration, system

# Create a single router to include all sub-routers
api_router = APIRouter()
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(jobs.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(configuration.router, prefix="/configuration", tags=["configuration"])
api_router.include_router(system.router, prefix="/system", tags=["system"])

__all__ = ["api_router"] 