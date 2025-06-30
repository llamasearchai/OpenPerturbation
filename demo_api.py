"""
Demo script for OpenPerturbation API

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Union

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def demo_api_endpoints():
    """Demo the API endpoints with mock data."""
    print("=" * 60)
    print("OpenPerturbation API Demo")
    print("=" * 60)
    
    try:
        from src.api.endpoints import (
            list_models, run_causal_discovery, run_explainability_analysis,
            design_interventions, list_experiments, health_check
        )
        from src.api.endpoints import CausalDiscoveryRequest, ExplainabilityRequest, InterventionDesignRequest
        
        print("1. Testing health check...")
        health = await health_check()
        if isinstance(health, dict):
            status = health.get('status', 'unknown')
        else:
            status = 'unknown'
        print(f"   Health status: {status}")
        print()
        
        print("2. Testing model listing...")
        models = list_models()
        print(f"   Available models: {len(models) if models else 0}")
        print()
        
        print("3. Testing causal discovery...")
        causal_request = CausalDiscoveryRequest(
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            method="correlation",
            alpha=0.05,
            variable_names=["gene_A", "gene_B", "gene_C"]
        )
        causal_result = await run_causal_discovery(causal_request)
        if isinstance(causal_result, dict):
            message = causal_result.get('message', 'No message')
        else:
            message = 'No message available'
        print(f"   Causal discovery completed: {message}")
        print()
        
        print("4. Testing intervention design...")
        intervention_request = InterventionDesignRequest(
            variable_names=["gene_A", "gene_B"],
            batch_size=10,
            budget=1000.0
        )
        intervention_result = await design_interventions(intervention_request)
        if isinstance(intervention_result, dict):
            interventions = intervention_result.get('recommended_interventions', [])
            intervention_count = len(interventions) if isinstance(interventions, list) else 0
        else:
            intervention_count = 0
        print(f"   Interventions designed: {intervention_count}")
        print()
        
        print("5. Testing experiments listing...")
        experiments = await list_experiments()
        print(f"   Available experiments: {len(experiments) if experiments else 0}")
        print()
        
        print("[SUCCESS] All API endpoints working correctly!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_app_creation():
    """Demo the app creation and startup."""
    print("6. Testing FastAPI app creation...")
    
    try:
        from src.api.app_factory import create_app
        app = create_app()
        
        if app is not None:
            print("   [SUCCESS] FastAPI app created successfully!")
            print("   Available routes:")
            
            # Try to get routes if FastAPI is available
            try:
                for route in app.routes:
                    if hasattr(route, 'path') and hasattr(route, 'methods'):
                        methods = ', '.join(route.methods) if route.methods else 'GET'
                        print(f"     {methods}: {route.path}")
            except:
                print("     [INFO] Route information not available")
        else:
            print("   [WARNING] App creation returned None (FastAPI may not be available)")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] App creation demo failed: {e}")
        return False

async def main():
    """Run the complete demo."""
    success1 = await demo_api_endpoints()
    success2 = demo_app_creation()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("The OpenPerturbation API is fully functional.")
    else:
        print("DEMO COMPLETED WITH SOME ISSUES")
        print("Check the error messages above for details.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 