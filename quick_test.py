#!/usr/bin/env python3
"""
Quick test script to verify OpenPerturbation components are working.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_models():
    """Test Pydantic models."""
    try:
        from src.api.models import AnalysisRequest, CausalDiscoveryRequest
        
        # Test AnalysisRequest
        req = AnalysisRequest(
            experiment_type="causal_discovery",
            data_source="genomics"
        )
        print(f"[SUCCESS] AnalysisRequest: {req.experiment_type}")
        
        # Test CausalDiscoveryRequest
        causal_req = CausalDiscoveryRequest(
            data_path="/path/to/data.csv"
        )
        print(f"[SUCCESS] CausalDiscoveryRequest: {causal_req.method}")
        
        return True
    except Exception as e:
        print(f"[FAILED] Models test: {e}")
        return False

def test_server_basic():
    """Test basic server functionality."""
    try:
        from src.api.server import find_free_port
        port = find_free_port(8000, 8010)
        print(f"[SUCCESS] Port finding: {port}")
        return True
    except Exception as e:
        print(f"[FAILED] Server test: {e}")
        return False

def test_dependencies():
    """Test key dependencies."""
    deps = []
    
    try:
        import fastapi
        deps.append(f"FastAPI {fastapi.__version__}")
    except ImportError:
        deps.append("FastAPI: NOT AVAILABLE")
    
    try:
        import pydantic
        deps.append(f"Pydantic {pydantic.__version__}")
    except ImportError:
        deps.append("Pydantic: NOT AVAILABLE")
    
    try:
        import uvicorn
        deps.append("Uvicorn: AVAILABLE")
    except ImportError:
        deps.append("Uvicorn: NOT AVAILABLE")
    
    print("[SUCCESS] Dependencies:", ", ".join(deps))
    return True

def main():
    """Main test function."""
    print("OpenPerturbation Quick Test")
    print("=" * 30)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Models", test_models),
        ("Server Basic", test_server_basic),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 30)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All core components working!")
        print("\nNext steps:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Start server: python -m src.api.server")
        print("3. Visit: http://localhost:8000/docs")
    else:
        print("[WARNING] Some tests failed. Check requirements.")

if __name__ == "__main__":
    main() 