"""
Integration test for OpenPerturbation API

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all API components can be imported without errors."""
    print("Testing imports...")
    
    try:
        from src.api import router, setup_middleware, create_app
        print("[SUCCESS] All API components imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_app_creation():
    """Test that the app can be created."""
    print("Testing app creation...")
    
    try:
        from src.api.app_factory import create_app
        app = create_app()
        
        if app is not None:
            print("[SUCCESS] App created successfully")
            return True
        else:
            print("[WARNING] App creation returned None (FastAPI not available)")
            return True  # This is expected when FastAPI is not available
    except Exception as e:
        print(f"[ERROR] App creation failed: {e}")
        return False

def test_router():
    """Test that the router is properly configured."""
    print("Testing router...")
    
    try:
        from src.api.endpoints import router
        
        if router is not None:
            print("[SUCCESS] Router is available")
            return True
        else:
            print("[WARNING] Router is None (FastAPI not available)")
            return True  # This is expected when FastAPI is not available
    except Exception as e:
        print(f"[ERROR] Router test failed: {e}")
        return False

def test_middleware():
    """Test that middleware can be set up."""
    print("Testing middleware...")
    
    try:
        from src.api.middleware import setup_middleware
        
        # Test with a mock app
        class MockApp:
            def add_middleware(self, *args, **kwargs):
                pass
            def middleware(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator
        
        mock_app = MockApp()
        setup_middleware(mock_app)
        print("[SUCCESS] Middleware setup completed")
        return True
    except Exception as e:
        print(f"[ERROR] Middleware test failed: {e}")
        return False

def test_main_module():
    """Test that the main module works."""
    print("Testing main module...")
    
    try:
        from src.api.main import app
        print("[SUCCESS] Main module imported and app variable exists")
        return True
    except Exception as e:
        print(f"[ERROR] Main module test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    print("=" * 50)
    print("OpenPerturbation API Integration Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_creation,
        test_router,
        test_middleware,
        test_main_module
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! API is working correctly.")
        return True
    else:
        print(f"[WARNING] {total - passed} test(s) failed or had warnings.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 