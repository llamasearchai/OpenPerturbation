from fastapi import APIRouter
from datetime import datetime
from typing import Dict
import platform, psutil

router = APIRouter()

@router.get("/info", response_model=Dict)
async def get_system_info():
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    } 