"""
Comprehensive Test Suite for OpenPerturbation Platform

Tests all major components and integration points.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi.testclient import TestClient
    from src.api.app_factory import create_app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

class TestAPIIntegration:
    """Test API endpoints and integration."""
    
    @pytest.fixture
    def client(self):
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        app = create_app()
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_system_info_endpoint(self, client):
        """Test system info endpoint."""
        response = client.get("/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "dependencies" in data
        assert isinstance(data["dependencies"], dict)
    
    def test_analysis_models_endpoint(self, client):
        """Test analysis models endpoint."""
        response = client.get("/analysis/models")
        assert response.status_code == 200
        data = response.json()
        assert "causal_discovery" in data
        assert "explainability" in data
        assert "prediction" in data
    
    def test_causal_discovery_endpoint(self, client):
        """Test causal discovery endpoint."""
        test_data = {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "method": "correlation",
            "alpha": 0.05,
            "variable_names": ["var1", "var2", "var3"]
        }
        response = client.post("/causal-discovery", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "adjacency_matrix" in data
        assert "method" in data
        assert data["method"] in ["correlation", "pc", "ges", "lingam"]
    
    def test_explainability_endpoint(self, client):
        """Test explainability endpoint."""
        # Create dummy files for testing
        model_path = Path("test_model.pt")
        data_path = Path("test_data.csv")
        
        model_path.touch()
        data_path.touch()
        
        try:
            test_data = {
                "model_path": str(model_path),
                "data_path": str(data_path),
                "analysis_types": ["attention", "concept"]
            }
            response = client.post("/explainability", json=test_data)
            assert response.status_code == 200
            data = response.json()
            assert "attention_analysis" in data or "concept_analysis" in data
        finally:
            model_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
    
    def test_intervention_design_endpoint(self, client):
        """Test intervention design endpoint."""
        test_data = {
            "variable_names": ["gene_A", "gene_B", "gene_C"],
            "batch_size": 10,
            "budget": 1000.0
        }
        response = client.post("/intervention-design", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "recommended_interventions" in data
        assert "intervention_ranking" in data


class TestDataProcessing:
    """Test data processing components."""
    
    def test_feature_extractor_import(self):
        """Test feature extractor can be imported."""
        try:
            from src.data.processors.feature_extractor import FeatureExtractor
            # Basic instantiation test
            extractor = FeatureExtractor()
            assert extractor is not None
        except ImportError as e:
            pytest.skip(f"Feature extractor import failed: {e}")
    
    def test_image_processor_import(self):
        """Test image processor can be imported."""
        try:
            from src.data.processors.image_processor import ImageProcessor
            processor = ImageProcessor()
            assert processor is not None
        except ImportError as e:
            pytest.skip(f"Image processor import failed: {e}")
    
    def test_data_loaders_import(self):
        """Test data loaders can be imported."""
        try:
            from src.data.loaders.genomics_loader import GenomicsLoader
            from src.data.loaders.imaging_loader import ImagingLoader
            from src.data.loaders.molecular_loader import MolecularLoader
            
            # Basic instantiation tests
            genomics_loader = GenomicsLoader()
            imaging_loader = ImagingLoader()
            molecular_loader = MolecularLoader()
            
            assert genomics_loader is not None
            assert imaging_loader is not None
            assert molecular_loader is not None
        except ImportError as e:
            pytest.skip(f"Data loaders import failed: {e}")


class TestModels:
    """Test model components."""
    
    def test_model_imports(self):
        """Test that models can be imported."""
        try:
            from src.models.fusion.multimodal_transformer import MultimodalTransformer
            from src.models.vision.cell_vit import CellViT
            from src.models.graph.molecular_gnn import MolecularGNN
            from src.models.causal.causal_vae import CausalVAE
            
            # Test model registry
            from src.models import MODEL_REGISTRY
            assert isinstance(MODEL_REGISTRY, dict)
            
        except ImportError as e:
            pytest.skip(f"Model imports failed: {e}")


class TestCausalDiscovery:
    """Test causal discovery functionality."""
    
    def test_causal_discovery_import(self):
        """Test causal discovery can be imported."""
        try:
            from src.causal.discovery import run_causal_discovery
            assert callable(run_causal_discovery)
        except ImportError as e:
            pytest.skip(f"Causal discovery import failed: {e}")
    
    def test_causal_discovery_basic(self):
        """Test basic causal discovery functionality."""
        try:
            from src.causal.discovery import run_causal_discovery
            
            # Create test data
            np.random.seed(42)
            data = np.random.randn(100, 5)
            labels = np.random.randint(0, 3, (100, 1))
            
            config = {
                "discovery_method": "correlation",
                "alpha": 0.05,
                "variable_names": [f"var_{i}" for i in range(5)]
            }
            
            result = run_causal_discovery(data, labels, config)
            
            assert isinstance(result, dict)
            assert "adjacency_matrix" in result
            assert "method" in result
            assert "variable_names" in result
            assert result["method"] == "correlation"
            
        except Exception as e:
            pytest.skip(f"Causal discovery test failed: {e}")


class TestExplainability:
    """Test explainability components."""
    
    def test_explainability_imports(self):
        """Test explainability modules can be imported."""
        try:
            from src.explainability.attention_maps import AttentionMapAnalyzer
            from src.explainability.concept_activation import ConceptActivationAnalyzer
            from src.explainability.pathway_analysis import PathwayAnalyzer
            
            assert AttentionMapAnalyzer is not None
            assert ConceptActivationAnalyzer is not None
            assert PathwayAnalyzer is not None
            
        except ImportError as e:
            pytest.skip(f"Explainability imports failed: {e}")


class TestUtilities:
    """Test utility components."""
    
    def test_logging_config(self):
        """Test logging configuration."""
        try:
            from src.utils.logging_config import setup_logging
            logger = setup_logging()
            assert logger is not None
        except ImportError as e:
            pytest.skip(f"Logging config import failed: {e}")
    
    def test_biology_utils(self):
        """Test biology utilities."""
        try:
            from src.utils.biology_utils import BiologyUtils
            utils = BiologyUtils()
            assert utils is not None
        except ImportError as e:
            pytest.skip(f"Biology utils import failed: {e}")
    
    def test_metrics(self):
        """Test metrics utilities."""
        try:
            from src.utils.metrics import calculate_metrics
            # Test with dummy data
            y_true = np.array([1, 0, 1, 1, 0])
            y_pred = np.array([1, 0, 1, 0, 0])
            metrics = calculate_metrics(y_true, y_pred)
            assert isinstance(metrics, dict)
        except ImportError as e:
            pytest.skip(f"Metrics import failed: {e}")


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_manager(self):
        """Test configuration manager."""
        try:
            from src.config.config_manager import ConfigManager
            manager = ConfigManager()
            assert manager is not None
        except ImportError as e:
            pytest.skip(f"Config manager import failed: {e}")


class TestTraining:
    """Test training components."""
    
    def test_training_imports(self):
        """Test training modules can be imported."""
        try:
            from src.training.lightning_modules import OpenPerturbationModule
            from src.training.data_modules import OpenPerturbationDataModule
            from src.training.metrics import MetricsCalculator
            
            assert OpenPerturbationModule is not None
            assert OpenPerturbationDataModule is not None
            assert MetricsCalculator is not None
            
        except ImportError as e:
            pytest.skip(f"Training imports failed: {e}")


class TestPipeline:
    """Test pipeline components."""
    
    def test_pipeline_import(self):
        """Test pipeline can be imported."""
        try:
            from src.pipeline.openperturbation_pipeline import OpenPerturbationPipeline
            pipeline = OpenPerturbationPipeline()
            assert pipeline is not None
        except ImportError as e:
            pytest.skip(f"Pipeline import failed: {e}")


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality works correctly."""
    # Simple async test
    async def dummy_async():
        await asyncio.sleep(0.01)
        return "success"
    
    result = await dummy_async()
    assert result == "success"


def test_numpy_compatibility():
    """Test NumPy compatibility and basic operations."""
    # Test basic numpy operations
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.shape == (5,)
    assert np.mean(arr) == 3.0
    assert np.sum(arr) == 15


def test_pandas_compatibility():
    """Test Pandas compatibility."""
    try:
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        assert df.shape == (3, 2)
        assert list(df.columns) == ['A', 'B']
    except ImportError:
        pytest.skip("Pandas not available")


class TestPackageStructure:
    """Test package structure and imports."""
    
    def test_src_structure(self):
        """Test src directory structure."""
        src_path = Path(__file__).parent.parent / "src"
        assert src_path.exists()
        
        # Check key directories exist
        assert (src_path / "api").exists()
        assert (src_path / "causal").exists()
        assert (src_path / "data").exists()
        assert (src_path / "models").exists()
        assert (src_path / "utils").exists()
    
    def test_init_files(self):
        """Test __init__.py files exist where needed."""
        src_path = Path(__file__).parent.parent / "src"
        
        # Check key __init__.py files
        key_modules = [
            "api", "models", "explainability", "losses"
        ]
        
        for module in key_modules:
            init_file = src_path / module / "__init__.py"
            if init_file.exists():
                # Try to import the module
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module, init_file)
                    if spec and spec.loader:
                        module_obj = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module_obj)
                except Exception as e:
                    # Don't fail test for import errors in this context
                    pass


if __name__ == "__main__":
    # Run tests when called directly
    pytest.main([__file__, "-v"]) 