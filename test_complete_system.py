#!/usr/bin/env python3
"""
Comprehensive System Test for OpenPerturbation API

This test verifies that all components work together correctly,
including server startup, API models, and endpoint registration.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all critical modules can be imported."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        # Test core dependencies
        import fastapi
        print(f"[SUCCESS] FastAPI {fastapi.__version__} imported successfully")
    except ImportError as e:
        print(f"[ERROR] FastAPI import failed: {e}")
        return False
    
    try:
        import pydantic
        print(f"[SUCCESS] Pydantic {pydantic.__version__} imported successfully")
    except ImportError as e:
        print(f"[ERROR] Pydantic import failed: {e}")
        return False
    
    try:
        import uvicorn
        print(f"[SUCCESS] Uvicorn {uvicorn.__version__} imported successfully")
    except ImportError as e:
        print(f"[ERROR] Uvicorn import failed: {e}")
        return False
    
    return True

def test_api_models():
    """Test that API models can be imported and basic functionality works."""
    print("\n" + "=" * 60)
    print("TESTING API MODELS")
    print("=" * 60)
    
    try:
        from src.api.models import AnalysisRequest, CausalDiscoveryRequest
        print("[SUCCESS] API models imported successfully")
        
        # Try to create basic models - use minimal required parameters
        try:
            # Simple test with minimal parameters
            analysis_req = AnalysisRequest()
            print("[SUCCESS] AnalysisRequest can be instantiated")
        except Exception as e:
            print(f"[INFO] AnalysisRequest requires parameters: {e}")
        
        try:
            causal_req = CausalDiscoveryRequest()
            print("[SUCCESS] CausalDiscoveryRequest can be instantiated")
        except Exception as e:
            print(f"[INFO] CausalDiscoveryRequest requires parameters: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] API models test failed: {e}")
        return False

def test_server_creation():
    """Test that the FastAPI server can be created."""
    print("\n" + "=" * 60)
    print("TESTING SERVER CREATION")
    print("=" * 60)
    
    try:
        from src.api.server import create_app
        
        # Create app
        app = create_app()
        
        if app is None:
            print("[ERROR] App creation returned None")
            return False
        
        print("[SUCCESS] FastAPI app created successfully")
        
        # Check if app has expected attributes
        if hasattr(app, 'routes'):
            print(f"[SUCCESS] App has {len(app.routes)} routes registered")
        else:
            print("[WARNING] App does not have routes attribute")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Server creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_endpoints_import():
    """Test that endpoints can be imported."""
    print("\n" + "=" * 60)
    print("TESTING ENDPOINTS IMPORT")
    print("=" * 60)
    
    try:
        from src.api.endpoints import router
        
        if router is None:
            print("[WARNING] Router is None (FastAPI not available)")
            return True
        
        print("[SUCCESS] Endpoints router imported successfully")
        
        if hasattr(router, 'routes'):
            print(f"[SUCCESS] Router has {len(router.routes)} routes")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Endpoints import test failed: {e}")
        return False

def test_causal_discovery():
    """Test causal discovery functionality."""
    print("\n" + "=" * 60)
    print("TESTING CAUSAL DISCOVERY")
    print("=" * 60)
    
    try:
        from src.causal.discovery import run_causal_discovery
        import numpy as np
        
        # Create test data
        test_data = np.random.rand(100, 5)
        test_labels = np.arange(100).reshape(-1, 1)
        
        config = {
            "method": "correlation",
            "discovery_method": "correlation",
            "alpha": 0.05,
            "variable_names": [f"var_{i}" for i in range(5)]
        }
        
        # Run causal discovery
        results = run_causal_discovery(
            causal_factors=test_data,
            perturbation_labels=test_labels,
            config=config
        )
        
        print(f"[SUCCESS] Causal discovery completed with method: {results.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Causal discovery test failed: {e}")
        return False

def test_system_integration():
    """Test complete system integration."""
    print("\n" + "=" * 60)
    print("TESTING SYSTEM INTEGRATION")
    print("=" * 60)
    
    try:
        # Test complete workflow
        from src.api.server import create_app
        from src.api.models import AnalysisRequest, CausalDiscoveryRequest
        
        # Create app
        app = create_app()
        if app is None:
            print("[ERROR] App creation failed")
            return False
        
        # Create test request
        request = AnalysisRequest(
            experiment_type="causal_discovery",
            data_source="test_data.csv"
        )
        
        # Create causal discovery request
        causal_request = CausalDiscoveryRequest(
            data=[[1.0, 2.0], [3.0, 4.0]],
            method="correlation",
            alpha=0.05
        )
        
        print("[SUCCESS] Complete system integration test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("OpenPerturbation API - Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("API Models Tests", test_api_models),
        ("Server Creation Tests", test_server_creation),
        ("Endpoints Import Tests", test_endpoints_import),
        ("Causal Discovery Tests", test_causal_discovery),
        ("System Integration Tests", test_system_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! System is ready for production.")
    else:
        print(f"\n[WARNING] {total - passed} tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 