"""
FastAPI server configuration and startup.
"""

import sys
import logging
import socket
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable, Protocol
from datetime import datetime
import os
import uuid
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Protocol definitions for type safety
class ASGIApp(Protocol):
    """Protocol for ASGI applications."""
    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        ...

class BackgroundTasksProtocol(Protocol):
    """Protocol for background tasks."""
    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        ...

class UploadFileProtocol(Protocol):
    """Protocol for upload files."""
    filename: str
    async def read(self) -> bytes:
        ...

# Type aliases for better type safety
ConfigType = Optional[Union[Dict[str, Any], Any]]  # Any for DictConfig when available

# Runtime availability flags
FASTAPI_AVAILABLE = False
UVICORN_AVAILABLE = False
OMEGACONF_AVAILABLE = False
MIDDLEWARE_AVAILABLE = False
ENDPOINTS_AVAILABLE = False

# Import FastAPI components with proper error handling
try:
    from fastapi import FastAPI
    from fastapi import BackgroundTasks
    from fastapi import UploadFile
    from fastapi import File as _File
    from fastapi import Form as _Form
    from fastapi.exceptions import HTTPException
    from fastapi.responses import RedirectResponse
    FASTAPI_AVAILABLE = True
    
except ImportError:
    logging.warning("FastAPI not available - server will run in stub mode")
    
    # Stub implementations
    class _BackgroundTasksStub:
        """Stub for BackgroundTasks when FastAPI is not available."""
        def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
            """Add task stub - does nothing."""
            pass

    class _UploadFileStub:
        """Stub for UploadFile when FastAPI is not available."""
        def __init__(self) -> None:
            self.filename: str = ""
        
        async def read(self) -> bytes:
            """Read stub - returns empty bytes."""
            return b""

    class _HTTPExceptionStub(Exception):
        """Stub for HTTPException when FastAPI is not available."""
        def __init__(self, status_code: int = 500, detail: Optional[str] = None):
            super().__init__(detail or "HTTP Exception")
            self.status_code = status_code
            self.detail = detail

    class _RouterStub:
        """Stub for FastAPI router."""
        def __init__(self):
            self.lifespan_context: Any = None

    class _FastAPIStub:
        """Stub for FastAPI when not available."""
        def __init__(self, *args: Any, **kwargs: Any):
            self.router = _RouterStub()

        def get(self, *args: Any, **kwargs: Any) -> Callable[[Callable], Callable]:
            """GET route decorator stub."""
            def decorator(func: Callable) -> Callable:
                return func
            return decorator

        def post(self, *args: Any, **kwargs: Any) -> Callable[[Callable], Callable]:
            """POST route decorator stub."""
            def decorator(func: Callable) -> Callable:
                return func
            return decorator

        def include_router(self, *args: Any, **kwargs: Any) -> None:
            """Include router stub."""
            pass

        def on_event(self, *args: Any, **kwargs: Any) -> Callable[[Callable], Callable]:
            """Event handler decorator stub."""
            def decorator(func: Callable) -> Callable:
                return func
            return decorator
            
        def add_event_handler(self, event: str, func: Callable) -> None:
            """Add event handler stub."""
            pass

    def _file_stub(*args: Any, **kwargs: Any) -> _UploadFileStub:
        """File parameter stub."""
        return _UploadFileStub()

    def _form_stub(*args: Any, **kwargs: Any) -> str:
        """Form parameter stub."""
        return ""

    # Assign stubs to names
    FastAPI = _FastAPIStub  # type: ignore[misc]
    BackgroundTasks = _BackgroundTasksStub  # type: ignore[misc]
    UploadFile = _UploadFileStub  # type: ignore[misc]
    HTTPException = _HTTPExceptionStub  # type: ignore[misc]
    _File = _file_stub  # type: ignore[misc]
    _Form = _form_stub  # type: ignore[misc]
    RedirectResponse = None  # type: ignore[misc]

# Import Uvicorn
try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    logging.warning("Uvicorn not available")
    uvicorn = None  # type: ignore[misc]

# Import OmegaConf
try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    logging.warning("OmegaConf not available")
    DictConfig = Dict[str, Any]  # type: ignore[misc]
    OmegaConf = None  # type: ignore[misc]

# Import middleware
try:
    from .middleware import setup_middleware
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    logging.warning("Middleware not available")
    def setup_middleware(app: Any) -> None:
        """Middleware setup stub."""
        pass

# Import endpoints
try:
    from .routes.analysis import router
    ENDPOINTS_AVAILABLE = True
except ImportError:
    logging.warning("Endpoints not available")
    router = None

logger = logging.getLogger(__name__)

def find_free_port(start_port: int = 8000, end_port: int = 8100) -> int:
    """Find a free port in the given range."""
    for port in range(start_port, end_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{end_port}")

def create_app(config: ConfigType = None) -> Optional[ASGIApp]:
    """Create and configure FastAPI application."""
    
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available, cannot create app")
        return None
    
    logger.info("Starting OpenPerturbation API Server...")
    
    app = FastAPI(
        title="OpenPerturbation API",
        description="API for perturbation biology analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Setup middleware
    if MIDDLEWARE_AVAILABLE:
        setup_middleware(app)
    else:
        logger.warning("Middleware not available, skipping setup")
    
    # Include routers
    if ENDPOINTS_AVAILABLE and router and FASTAPI_AVAILABLE:
        try:
            app.include_router(router, prefix="/api/v1")  # type: ignore[arg-type]
            logger.info("API endpoints registered successfully") 
        except Exception as e:
            logger.warning(f"Could not include router: {e}")
    else:
        logger.warning("Endpoints not available, API will have limited functionality")
    
    @app.get("/", response_model=dict)
    async def root() -> Dict[str, Any]:
        """Root endpoint returning JSON instead of redirecting to docs."""
        return {
            "service": "OpenPerturbation API",
            "version": "1.0.0",
            "status": "running",
            "documentation": "/docs",
            "health_check": "/health",
            "logo": "/logo",
            "author": "Nik Jois",
            "email": "nikjois@llamasearch.ai"
        }
    
    @app.get("/logo")
    async def get_logo():
        """Serve the OpenPerturbation logo."""
        try:
            from fastapi.responses import FileResponse
            logo_path = project_root / "Logo.svg"
            if logo_path.exists():
                return FileResponse(
                    path=str(logo_path),
                    media_type="image/svg+xml",
                    filename="openperturbation-logo.svg"
                )
            else:
                return {"error": "Logo file not found"}
        except ImportError:
            return {"error": "FileResponse not available"}
    
    @app.get("/health", response_model=dict)
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint with timestamp."""
        return {
            "service": "OpenPerturbation API",
            "status": "healthy", 
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def startup_event() -> None:
        """Startup event handler."""
        logger.info("OpenPerturbation API Server started successfully")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU")
        except ImportError:
            logger.info("PyTorch not available - using CPU")
    
    async def shutdown_event() -> None:
        """Shutdown event handler."""
        logger.info("Shutting down OpenPerturbation API Server...")
    
    # Add lifespan context manager for modern FastAPI
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def lifespan(app_instance):
        # Startup
        await startup_event()
        yield
        # Shutdown
        await shutdown_event()
    
    # Use lifespan if FastAPI supports it
    try:
        app.router.lifespan_context = lifespan
    except AttributeError:
        # Fallback to deprecated on_event for older FastAPI versions
        app.add_event_handler("startup", startup_event)
        app.add_event_handler("shutdown", shutdown_event)
    
    # Analysis endpoints
    @app.post("/api/v1/analysis/start", response_model=dict)
    async def start_analysis(
        request: Dict[str, Any], background_tasks: Any
    ) -> Dict[str, Any]:
        """Start analysis job."""
        try:
            job_id = str(uuid.uuid4())
            
            # Store job in background tasks
            background_tasks.add_task(
                run_analysis_pipeline,
                job_id=job_id,
                config=request
            )
            
            analysis_jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat(),
                "config": request,
                "progress": 0,
                "message": "Analysis job queued"
            }
            
            return {
                "job_id": job_id,
                "status": "queued",
                "message": "Analysis started successfully"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/analysis/status/{job_id}", response_model=dict)
    async def get_analysis_status(job_id: str) -> Dict[str, Any]:
        """Get analysis job status."""
        if job_id not in analysis_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return analysis_jobs[job_id]

    # Data upload endpoint
    @app.post("/api/v1/data/upload", response_model=dict)
    async def upload_data(
        file: Any = _File(...),
        data_type: str = _Form(...)
    ) -> Dict[str, Any]:
        """Upload data file."""
        # Validate file type
        allowed_types = {
            'genomics': ['.csv', '.tsv', '.h5', '.xlsx'],
            'imaging': ['.png', '.jpg', '.jpeg', '.tiff', '.tif'],
            'chemical': ['.sdf', '.mol', '.csv', '.tsv']
        }
        
        file_extension = Path(file.filename).suffix.lower()
        
        if data_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported data type: {data_type}")
        
        if file_extension not in allowed_types[data_type]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type {file_extension} for {data_type} data"
            )
        
        # Save file
        upload_dir = Path("uploads") / data_type
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            return {
                "filename": file.filename,
                "data_type": data_type,
                "file_size": len(content),
                "file_path": str(file_path),
                "upload_timestamp": datetime.utcnow().isoformat(),
                "status": "uploaded"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    @app.get("/api/v1/models", response_model=List[dict])
    async def list_available_models() -> List[Dict[str, Any]]:
        """List available models."""
        models = [
            {
                "name": "multimodal_fusion",
                "description": "Multimodal fusion model for integrating different data types",
                "version": "1.0",
                "input_types": ["genomics", "imaging", "chemical"],
                "parameters": {
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "dropout": 0.1
                }
            },
            {
                "name": "causal_vae", 
                "description": "Causal Variational Autoencoder for causal discovery",
                "version": "1.0",
                "input_types": ["genomics"],
                "parameters": {
                    "latent_dim": 128,
                    "beta": 1.0,
                    "num_layers": 4
                }
            },
            {
                "name": "cell_vit",
                "description": "Vision Transformer for cellular imaging analysis", 
                "version": "1.0",
                "input_types": ["imaging"],
                "parameters": {
                    "patch_size": 16,
                    "embed_dim": 768,
                    "num_heads": 12,
                    "num_layers": 12
                }
            },
            {
                "name": "causal_discovery",
                "description": "Causal discovery engine using various algorithms",
                "version": "1.0", 
                "input_types": ["genomics", "chemical"],
                "parameters": {
                    "method": "pc",
                    "alpha": 0.05,
                    "max_depth": 3
                }
            }
        ]
        return models

    @app.post("/api/v1/intervention-design", response_model=dict)
    async def design_interventions(request: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal interventions."""
        try:
            variable_names = request.get("variable_names", ["gene_A", "gene_B"])
            batch_size = request.get("batch_size", 10)
            budget = request.get("budget", 1000.0)
            
            # Mock intervention design logic
            interventions = [
                {
                    "target": variable_names[0] if variable_names else "gene_A",
                    "type": "knockout",
                    "confidence": 0.95,
                    "expected_cost": 150.0
                },
                {
                    "target": variable_names[1] if len(variable_names) > 1 else "gene_B", 
                    "type": "overexpression",
                    "confidence": 0.87,
                    "expected_cost": 200.0
                }
            ]
            
            # Calculate expected effects
            expected_effects = {}
            for var in variable_names[:2]:
                expected_effects[var] = {
                    "phenotype_1": round(random.uniform(0.3, 1.0), 2),
                    "phenotype_2": round(random.uniform(-0.5, 0.5), 2)
                }
            
            total_cost = sum(i["expected_cost"] for i in interventions)
            
            return {
                "interventions_designed": len(interventions),
                "interventions": interventions,
                "expected_effects": expected_effects,
                "total_estimated_cost": total_cost,
                "design_confidence": 0.91,
                "batch_size": batch_size,
                "within_budget": total_cost <= budget
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/experiments", response_model=List[dict])
    async def list_experiments() -> List[Dict[str, Any]]:
        """List experiment configurations."""
        experiments = [
            {
                "id": "exp_001",
                "name": "High Content Screening",
                "description": "Cell morphology analysis with compound perturbations",
                "data_sources": ["imaging", "chemical"],
                "status": "active",
                "created_at": "2024-01-01T10:00:00Z",
                "config": {
                    "model_type": "cell_vit",
                    "batch_size": 32,
                    "num_epochs": 100
                }
            },
            {
                "id": "exp_002", 
                "name": "Genomics Perturbation",
                "description": "Gene expression analysis with genetic perturbations",
                "data_sources": ["genomics"],
                "status": "completed",
                "created_at": "2024-01-01T11:00:00Z", 
                "config": {
                    "model_type": "causal_vae",
                    "batch_size": 64,
                    "num_epochs": 50
                }
            }
        ]
        return experiments

    @app.get("/api/v1/datasets", response_model=List[dict])
    async def list_datasets() -> List[Dict[str, Any]]:
        """List available datasets."""
        datasets = [
            {
                "name": "ChEMBL_compounds",
                "description": "Chemical database of bioactive molecules",
                "data_type": "chemical",
                "format": "CSV",
                "size": 2000000,
                "source": "https://www.ebi.ac.uk/chembl/",
                "last_updated": "2024-01-01"
            },
            {
                "name": "TCGA_expression",
                "description": "Cancer genomics expression data",
                "data_type": "genomics",
                "format": "H5",
                "size": 500000,
                "source": "https://www.cancer.gov/tcga",
                "last_updated": "2023-12-15"
            },
            {
                "name": "Cell_Painting",
                "description": "High-content imaging dataset", 
                "data_type": "imaging",
                "format": "PNG",
                "size": 1000000,
                "source": "Broad Institute",
                "last_updated": "2024-01-10"
            }
        ]
        return datasets

    @app.post("/api/v1/validate-config", response_model=dict)
    async def validate_configuration(config: dict) -> Dict[str, Any]:
        """Validate configuration."""
        errors = []
        warnings = []
        
        # Required fields validation
        required_fields = ["experiment_type", "data_source"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Warning checks
        if "model" in config:
            model_config = config["model"]
            if model_config.get("batch_size", 32) > 128:
                warnings.append("Large batch size may cause memory issues")
            if model_config.get("learning_rate", 0.001) > 0.01:
                warnings.append("High learning rate may cause training instability")
        
        if "data" in config:
            data_config = config["data"]
            if data_config.get("num_workers", 4) > 8:
                warnings.append("High number of workers may not improve performance")
        
        # Performance warnings
        if config.get("max_epochs", 100) > 1000:
            warnings.append("Very high epoch count - consider early stopping")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    @app.get("/api/v1/system/info", response_model=dict)
    async def get_system_info() -> Dict[str, Any]:
        """Get system information."""
        try:
            import torch
            pytorch_version = torch.__version__
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pytorch_version = "not available"
            gpu_available = False
        
        # Check dependency availability
        dependencies = {}
        try:
            import pandas
            dependencies["pandas"] = True
        except ImportError:
            dependencies["pandas"] = False
        
        dependencies["fastapi"] = FASTAPI_AVAILABLE
        dependencies["omegaconf"] = OMEGACONF_AVAILABLE
        
        try:
            import causal_learn  # type: ignore[import-not-found]
            dependencies["causal"] = True
        except ImportError:
            dependencies["causal"] = False
        
        try:
            import platform
            import psutil
            return {
                "python_version": platform.python_version(),
                "pytorch_version": pytorch_version,
                "platform": platform.system(),
                "cpu_count": psutil.cpu_count(),
                "memory_available": psutil.virtual_memory().total,
                "gpu_available": gpu_available,
                "dependencies": dependencies
            }
        except ImportError:
            return {
                "python_version": "unknown",
                "pytorch_version": pytorch_version,
                "platform": "unknown",
                "cpu_count": "unknown",
                "memory_available": "unknown",
                "gpu_available": gpu_available,
                "dependencies": dependencies
            }

    return app  # type: ignore[return-value]

def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    config_path: Optional[str] = None,
    auto_find_port: bool = True
) -> None:
    """Run the FastAPI server."""
    
    if not UVICORN_AVAILABLE:
        logger.error("Uvicorn not available, cannot run server")
        return
    
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available, cannot run server")
        return
    
    # Load configuration
    config = None
    if config_path and OMEGACONF_AVAILABLE and OmegaConf:
        try:
            config = OmegaConf.load(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    # Create app
    app = create_app(config)
    if app is None:
        logger.error("Could not create FastAPI app")
        return
    
    # Find available port if requested
    if auto_find_port:
        try:
            port = find_free_port(port, port + 100)
            logger.info(f"Using port {port}")
        except RuntimeError as e:
            logger.error(str(e))
            return
    
    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "log_level": "info",
        "access_log": True,
        "reload": False,  # Set to True for development
    }
    
    try:
        # Run server
        logger.info(f"Starting server on {host}:{port}")
        if uvicorn:
            uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

# Global app instance for use with ASGI servers
app = create_app()

# Background server setup for testing
if UVICORN_AVAILABLE and FASTAPI_AVAILABLE and app is not None:
    import threading

    def _background_server() -> None:
        """Background server for testing."""
        if uvicorn and app:
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error", access_log=False)

    _server_thread = threading.Thread(target=_background_server, daemon=True)
    _server_thread.start()

def main() -> None:
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenPerturbation API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-auto-port", action="store_true", help="Disable automatic port finding")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        auto_find_port=not args.no_auto_port
    )

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# In-memory job registry & background task implementation
# ----------------------------------------------------------------------------

analysis_jobs: Dict[str, Dict[str, Any]] = {}

def run_analysis_pipeline(job_id: str, config: Dict[str, Any]) -> None:
    """Mock implementation that marks a job complete after brief delay."""
    import time

    try:
        for _ in range(2):
            time.sleep(0.01)
            if job_id in analysis_jobs:
                analysis_jobs[job_id]["progress"] += 50
        
        if job_id in analysis_jobs:
            analysis_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "results": {"summary": "Mock analysis complete"},
                "completed_at": datetime.utcnow().isoformat(),
            })
    except Exception as e:
        if job_id in analysis_jobs:
            analysis_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat(),
            })
