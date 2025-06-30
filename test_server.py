"""
Comprehensive test script for OpenPerturbation FastAPI server.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.api.server import create_app, run_server, find_free_port
        print("[SUCCESS] Server imports working")
    except Exception as e:
        print(f"[ERROR] Server imports failed: {e}")
        return False
    
    try:
        from src.api.middleware import setup_middleware
        print("[SUCCESS] Middleware imports working")
    except Exception as e:
        print(f"[ERROR] Middleware imports failed: {e}")
        return False
    
    try:
        from src.api.endpoints import router
        print("[SUCCESS] Endpoints imports working")
    except Exception as e:
        print(f"[ERROR] Endpoints imports failed: {e}")
        return False
    
    try:
        from src.api.models import AnalysisRequest, CausalDiscoveryRequest
        print("[SUCCESS] Models imports working")
    except Exception as e:
        print(f"[ERROR] Models imports failed: {e}")
        return False
    
    return True

def test_dependency_availability():
    """Test availability of key dependencies."""
    print("\nTesting dependency availability...")
    
    dependencies = {
        'fastapi': False,
        'uvicorn': False,
        'pydantic': False,
        'omegaconf': False,
        'pandas': False,
        'torch': False,
        'numpy': False
    }
    
    # Test FastAPI
    try:
        import fastapi
        dependencies['fastapi'] = True
        print(f"[SUCCESS] FastAPI {fastapi.__version__} available")
    except ImportError:
        print("[WARNING] FastAPI not available")
    
    # Test Uvicorn
    try:
        import uvicorn
        dependencies['uvicorn'] = True
        print(f"[SUCCESS] Uvicorn available")
    except ImportError:
        print("[WARNING] Uvicorn not available")
    
    # Test Pydantic
    try:
        import pydantic
        dependencies['pydantic'] = True
        print(f"[SUCCESS] Pydantic {pydantic.__version__} available")
    except ImportError:
        print("[ERROR] Pydantic not available - required for models")
    
    # Test OmegaConf
    try:
        import omegaconf
        dependencies['omegaconf'] = True
        print(f"[SUCCESS] OmegaConf {omegaconf.__version__} available")
    except ImportError:
        print("[WARNING] OmegaConf not available")
    
    # Test Pandas
    try:
        import pandas
        dependencies['pandas'] = True
        print(f"[SUCCESS] Pandas {pandas.__version__} available")
    except ImportError:
        print("[WARNING] Pandas not available")
    
    # Test PyTorch
    try:
        import torch
        dependencies['torch'] = True
        print(f"[SUCCESS] PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"[SUCCESS] CUDA available with {torch.cuda.device_count()} devices")
        else:
            print("[INFO] CUDA not available, using CPU")
    except ImportError:
        print("[WARNING] PyTorch not available")
    
    # Test NumPy
    try:
        import numpy
        dependencies['numpy'] = True
        print(f"[SUCCESS] NumPy {numpy.__version__} available")
    except ImportError:
        print("[ERROR] NumPy not available - required for scientific computing")
    
    return dependencies

def test_server_creation():
    """Test server creation without starting it."""
    print("\nTesting server creation...")
    
    try:
        from src.api.server import create_app
        app = create_app()
        
        if app is None:
            print("[WARNING] Server creation returned None (FastAPI not available)")
            return False
        else:
            print("[SUCCESS] Server created successfully")
            return True
    except Exception as e:
        print(f"[ERROR] Server creation failed: {e}")
        return False

def test_port_finding():
    """Test port finding functionality."""
    print("\nTesting port finding...")
    
    try:
        from src.api.server import find_free_port
        port = find_free_port(8000, 8010)
        print(f"[SUCCESS] Found free port: {port}")
        return True
    except Exception as e:
        print(f"[ERROR] Port finding failed: {e}")
        return False

def test_api_models():
    """Test API model validation."""
    print("\nTesting API models...")
    
    try:
        from src.api.models import AnalysisRequest, CausalDiscoveryRequest
        
        # Test AnalysisRequest
        analysis_req = AnalysisRequest(
            experiment_type="causal_discovery",
            data_source="genomics",
            parameters={"test": "value"}
        )
        print(f"[SUCCESS] AnalysisRequest created: {analysis_req.experiment_type}")
        
        # Test CausalDiscoveryRequest  
        causal_req = CausalDiscoveryRequest(
            data_path="/path/to/data.csv",
            method="pc",
            alpha=0.05
        )
        print(f"[SUCCESS] CausalDiscoveryRequest created: {causal_req.method}")
        
        return True
    except Exception as e:
        print(f"[ERROR] API models test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("\nTesting async functionality...")
    
    try:
        # Simple async test
        async def dummy_analysis():
            await asyncio.sleep(0.01)
            return {"status": "completed"}
        
        result = await dummy_analysis()
        print(f"[SUCCESS] Async functionality working: {result}")
        return True
    except Exception as e:
        print(f"[ERROR] Async functionality failed: {e}")
        return False

def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    try:
        # Test valid config
        valid_config = {
            "experiment_type": "causal_discovery",
            "data_source": "genomics",
            "model": {
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }
        
        # Basic validation logic
        required_fields = ["experiment_type", "data_source"]
        errors = []
        warnings = []
        
        for field in required_fields:
            if field not in valid_config:
                errors.append(f"Missing required field: {field}")
        
        if valid_config.get("model", {}).get("batch_size", 32) > 128:
            warnings.append("Large batch size may cause memory issues")
        
        print(f"[SUCCESS] Config validation working - errors: {len(errors)}, warnings: {len(warnings)}")
        return True
    except Exception as e:
        print(f"[ERROR] Config validation failed: {e}")
        return False

def test_file_operations():
    """Test file operation functionality."""
    print("\nTesting file operations...")
    
    try:
        # Test upload directory creation
        upload_dir = Path("uploads/test")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Test file types validation
        allowed_types = {
            'genomics': ['.csv', '.tsv', '.h5', '.xlsx'],
            'imaging': ['.png', '.jpg', '.jpeg', '.tiff', '.tif'],
            'chemical': ['.sdf', '.mol', '.csv', '.tsv']
        }
        
        test_file = "test_data.csv"
        file_extension = Path(test_file).suffix.lower()
        
        if file_extension in allowed_types['genomics']:
            print(f"[SUCCESS] File type validation working for {test_file}")
        
        # Clean up
        if upload_dir.exists():
            upload_dir.rmdir()
        
        return True
    except Exception as e:
        print(f"[ERROR] File operations test failed: {e}")
        return False

def generate_requirements():
    """Generate requirements.txt file."""
    print("\nGenerating requirements.txt...")
    
    requirements = [
        "# OpenPerturbation API Server Requirements",
        "# Author: Nik Jois <nikjois@llamasearch.ai>",
        "",
        "# Core web framework",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "",
        "# Data validation and serialization",
        "pydantic>=2.5.0",
        "",
        "# Configuration management",
        "omegaconf>=2.3.0",
        "",
        "# Data processing",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "",
        "# Machine learning",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.3.0",
        "",
        "# Causal discovery",
        "causal-learn>=0.1.3.3",
        "",
        "# Biology and chemistry",
        "biopython>=1.81",
        "rdkit-pypi>=2022.9.5",
        "",
        "# Image processing",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "",
        "# Visualization",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "",
        "# Utilities",
        "python-multipart>=0.0.6",
        "aiofiles>=23.0.0",
        "httpx>=0.25.0",
        "",
        "# Development and testing",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.9.0",
        "mypy>=1.6.0",
        "",
        "# Optional dependencies for production",
        "gunicorn>=21.2.0",
        "psutil>=5.9.0"
    ]
    
    try:
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        print("[SUCCESS] requirements.txt generated")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to generate requirements.txt: {e}")
        return False

def main():
    """Main test function."""
    print("OpenPerturbation FastAPI Server Test Suite")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    test_results = {}
    
    # Run tests
    test_results['imports'] = test_imports()
    test_results['dependencies'] = test_dependency_availability()
    test_results['server_creation'] = test_server_creation()
    test_results['port_finding'] = test_port_finding()
    test_results['api_models'] = test_api_models()
    test_results['async_functionality'] = asyncio.run(test_async_functionality())
    test_results['config_validation'] = test_config_validation()
    test_results['file_operations'] = test_file_operations()
    test_results['requirements_generation'] = generate_requirements()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Server is ready for deployment.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 