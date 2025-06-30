"""
Test runner script for OpenPerturbation.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

def run_command(cmd: List[str], description: str, capture_output: bool = True) -> bool:
    """Run a command and return success status."""
    print(f"\n[RUNNING] {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=True)
        
        print(f"[SUCCESS] {description}")
        if capture_output and result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"Error code: {e.returncode}")
        if capture_output:
            if e.stdout:
                print("Stdout:")
                print(e.stdout)
            if e.stderr:
                print("Stderr:")
                print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"[FAILED] {description} - Command not found")
        return False

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {
        "pytest": False,
        "pytest-asyncio": False,
        "pytest-cov": False,
        "httpx": False,
        "black": False,
        "isort": False,
        "flake8": False,
        "mypy": False
    }
    
    # Check pytest
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                     capture_output=True, check=True)
        dependencies["pytest"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check pytest-asyncio
    try:
        import pytest_asyncio
        dependencies["pytest-asyncio"] = True
    except ImportError:
        pass
    
    # Check pytest-cov
    try:
        import pytest_cov
        dependencies["pytest-cov"] = True
    except ImportError:
        pass
    
    # Check httpx
    try:
        import httpx
        dependencies["httpx"] = True
    except ImportError:
        pass
    
    # Check other tools
    for dep in ["black", "isort", "flake8", "mypy"]:
        try:
            subprocess.run([sys.executable, "-m", dep, "--version"], 
                         capture_output=True, check=True)
            dependencies[dep] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return dependencies

def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Running unit tests")

def run_integration_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Running integration tests")

def run_api_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run API tests."""
    cmd = ["python", "-m", "pytest", "tests/test_api.py"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Running API tests")

def run_all_tests(verbose: bool = False, coverage: bool = False, fast: bool = False) -> bool:
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if fast:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "Running all tests")

def run_linting() -> bool:
    """Run code linting."""
    success = True
    
    # Run flake8
    flake8_success = run_command(
        ["python", "-m", "flake8", "src", "tests"],
        "Running flake8 linting"
    )
    success = success and flake8_success
    
    return success

def run_type_checking() -> bool:
    """Run type checking."""
    return run_command(
        ["python", "-m", "mypy", "src"],
        "Running mypy type checking"
    )

def run_format_checking() -> bool:
    """Run code format checking."""
    success = True
    
    # Check black formatting
    black_success = run_command(
        ["python", "-m", "black", "--check", "src", "tests"],
        "Checking code formatting with black"
    )
    success = success and black_success
    
    # Check import sorting
    isort_success = run_command(
        ["python", "-m", "isort", "--check-only", "src", "tests"],
        "Checking import sorting with isort"
    )
    success = success and isort_success
    
    return success

def run_security_checks() -> bool:
    """Run security checks."""
    success = True
    
    # Run bandit for security issues
    try:
        bandit_success = run_command(
            ["python", "-m", "bandit", "-r", "src"],
            "Running security checks with bandit"
        )
        success = success and bandit_success
    except FileNotFoundError:
        print("[SKIPPED] bandit not available - skipping security checks")
    
    return success

def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate a test report."""
    print("\n" + "="*60)
    print("TEST REPORT")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_type, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_type}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("COMPLETE: All tests passed!")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run OpenPerturbation tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--format-check", action="store_true", help="Run format checking")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--fix", action="store_true", help="Fix formatting issues")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        print("Checking dependencies...")
        deps = check_dependencies()
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            print(f"{status} {dep}")
        return
    
    # Fix formatting if requested
    if args.fix:
        print("Fixing code formatting...")
        run_command(["python", "-m", "black", "src", "tests"], "Fixing code formatting with black", capture_output=False)
        run_command(["python", "-m", "isort", "src", "tests"], "Fixing import sorting with isort", capture_output=False)
        return
    
    # Default to running all tests if no specific test type is specified
    if not any([args.unit, args.integration, args.api, args.all, args.lint, args.type_check, args.format_check, args.security]):
        args.all = True
    
    results = {}
    
    # Run tests
    if args.unit:
        results["Unit Tests"] = run_unit_tests(args.verbose, args.coverage)
    
    if args.integration:
        results["Integration Tests"] = run_integration_tests(args.verbose, args.coverage)
    
    if args.api:
        results["API Tests"] = run_api_tests(args.verbose, args.coverage)
    
    if args.all:
        results["All Tests"] = run_all_tests(args.verbose, args.coverage, args.fast)
    
    # Run additional checks
    if args.lint:
        results["Linting"] = run_linting()
    
    if args.type_check:
        results["Type Checking"] = run_type_checking()
    
    if args.format_check:
        results["Format Checking"] = run_format_checking()
    
    if args.security:
        results["Security Checks"] = run_security_checks()
    
    # Generate report
    if results:
        generate_test_report(results)
    else:
        print("No tests or checks specified. Use --help for options.")

if __name__ == "__main__":
    main() 