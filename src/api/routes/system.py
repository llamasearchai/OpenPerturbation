from fastapi import APIRouter
from datetime import datetime, timezone
from typing import Dict
import platform, psutil
import sys
import importlib.util
import subprocess

router = APIRouter()

def get_package_version(package_name: str) -> str:
    """Get the version of an installed package."""
    try:
        # First try using pkg_resources which is safer
        import pkg_resources
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            pass
    except ImportError:
        pass
    
    # Fallback to subprocess pip show (safer than importing)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version: '):
                    return line.split(': ')[1].strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Final fallback - direct import (risky but last resort)
    try:
        package = importlib.import_module(package_name)
        return getattr(package, '__version__', 'unknown')
    except (ImportError, AssertionError, Exception):
        return 'not installed'

@router.get("/info", response_model=Dict)
async def get_system_info():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "dependencies": {
            "fastapi": get_package_version("fastapi"),
            "uvicorn": get_package_version("uvicorn"),
            "pytorch": get_package_version("torch"),
            "pytorch_lightning": get_package_version("lightning"),  # Changed to 'lightning' as modern name
            "numpy": get_package_version("numpy"),
            "pandas": get_package_version("pandas"),
            "scipy": get_package_version("scipy"),
            "scikit-learn": get_package_version("scikit-learn"),
            "transformers": get_package_version("transformers"),
            "matplotlib": get_package_version("matplotlib"),
            "seaborn": get_package_version("seaborn"),
            "plotly": get_package_version("plotly"),
            "networkx": get_package_version("networkx"),
            "torchmetrics": get_package_version("torchmetrics"),
            "scanpy": get_package_version("scanpy"),
            "psutil": get_package_version("psutil"),
        }
    } 