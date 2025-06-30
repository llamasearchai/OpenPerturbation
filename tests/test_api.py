"""
Comprehensive test suite for OpenPerturbation API.
"""

import pytest
import httpx
import asyncio
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock

# Test data
SAMPLE_ANALYSIS_REQUEST = {
    "experiment_type": "causal_discovery",
    "data_source": "genomics",
    "model_config": {
        "method": "pc",
        "alpha": 0.05
    },
    "parameters": {
        "max_depth": 3
    }
}

SAMPLE_INTERVENTION_REQUEST = {
    "variable_names": ["gene_A", "gene_B", "gene_C"],
    "budget": 1000.0,
    "batch_size": 10,
    "optimization_target": "maximize_effect"
}

SAMPLE_CONFIG = {
    "experiment_type": "high_content_screening",
    "data_source": "imaging",
    "model": {
        "type": "cell_vit",
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "data": {
        "num_workers": 4
    },
    "max_epochs": 100
}

@pytest.fixture
async def client():
    """Create test client."""
    from src.api.server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("gene,expression\n")
        f.write("gene_A,1.2\n")
        f.write("gene_B,0.8\n")
        f.write("gene_C,1.5\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)

class TestHealthEndpoints:
    """Test health and root endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "OpenPerturbation API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "OpenPerturbation API"
        assert data["status"] == "healthy"
        assert "timestamp" in data

class TestAnalysisEndpoints:
    """Test analysis-related endpoints."""
    
    def test_start_analysis(self, client):
        """Test starting an analysis job."""
        response = client.post("/api/v1/analysis/start", json=SAMPLE_ANALYSIS_REQUEST)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["message"] == "Analysis started successfully"
    
    def test_get_analysis_status(self, client):
        """Test getting analysis status."""
        # First start an analysis
        start_response = client.post("/api/v1/analysis/start", json=SAMPLE_ANALYSIS_REQUEST)
        job_id = start_response.json()["job_id"]
        
        # Then check status
        response = client.get(f"/api/v1/analysis/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["status"] == "queued"
    
    def test_get_nonexistent_analysis_status(self, client):
        """Test getting status for non-existent job."""
        response = client.get("/api/v1/analysis/status/nonexistent")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

class TestCausalDiscoveryEndpoints:
    """Test causal discovery endpoints."""
    
    def test_causal_discovery(self, client):
        """Test causal discovery endpoint."""
        request_data = {
            "data_path": "/path/to/data.csv",
            "method": "pc",
            "alpha": 0.05,
            "max_depth": 3,
            "variables": ["gene_A", "gene_B", "gene_C"]
        }
        
        response = client.post("/api/v1/causal/discovery", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "graph" in data
        assert "metrics" in data
        assert data["method_used"] == "pc"
        assert data["status"] == "completed"

class TestInterventionDesignEndpoints:
    """Test intervention design endpoints."""
    
    def test_design_interventions(self, client):
        """Test intervention design endpoint."""
        response = client.post("/api/v1/intervention/design", json=SAMPLE_INTERVENTION_REQUEST)
        assert response.status_code == 200
        data = response.json()
        assert "interventions_designed" in data
        assert "interventions" in data
        assert "expected_effects" in data
        assert "total_estimated_cost" in data
        assert "design_confidence" in data
        assert "within_budget" in data

class TestExplainabilityEndpoints:
    """Test explainability endpoints."""
    
    def test_analyze_explainability(self, client):
        """Test explainability analysis endpoint."""
        request_data = {
            "model_path": "/path/to/model.pth",
            "data_path": "/path/to/data.csv",
            "method": "attention",
            "target_variables": ["gene_A", "gene_B"]
        }
        
        response = client.post("/api/v1/explainability/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "explanations" in data
        assert "importance_scores" in data
        assert data["method_used"] == "attention"
        assert data["status"] == "completed"

class TestDataUploadEndpoints:
    """Test data upload endpoints."""
    
    def test_upload_genomics_data(self, client, sample_csv_file):
        """Test uploading genomics data."""
        with open(sample_csv_file, 'rb') as f:
            files = {'file': ('test.csv', f, 'text/csv')}
            data = {'data_type': 'genomics'}
            response = client.post("/api/v1/data/upload", files=files, data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.csv"
        assert data["data_type"] == "genomics"
        assert data["status"] == "uploaded"
        assert "file_id" in data
    
    def test_upload_invalid_file_type(self, client):
        """Test uploading invalid file type."""
        files = {'file': ('test.txt', b'invalid content', 'text/plain')}
        data = {'data_type': 'genomics'}
        response = client.post("/api/v1/data/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_upload_invalid_data_type(self, client):
        """Test uploading with invalid data type."""
        files = {'file': ('test.csv', b'content', 'text/csv')}
        data = {'data_type': 'invalid_type'}
        response = client.post("/api/v1/data/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "Unsupported data type" in response.json()["detail"]

class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check model structure
        model = models[0]
        assert "name" in model
        assert "description" in model
        assert "version" in model
        assert "input_types" in model
        assert "parameters" in model
    
    def test_get_model_info(self, client):
        """Test getting specific model information."""
        response = client.get("/api/v1/models/multimodal_fusion")
        assert response.status_code == 200
        model = response.json()
        assert model["name"] == "multimodal_fusion"
        assert "multimodal fusion" in model["description"].lower()
    
    def test_get_nonexistent_model(self, client):
        """Test getting non-existent model."""
        response = client.get("/api/v1/models/nonexistent")
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]

class TestExperimentEndpoints:
    """Test experiment management endpoints."""
    
    def test_list_experiments(self, client):
        """Test listing experiments."""
        response = client.get("/api/v1/experiments")
        assert response.status_code == 200
        experiments = response.json()
        assert isinstance(experiments, list)
        assert len(experiments) > 0
        
        # Check experiment structure
        experiment = experiments[0]
        assert "id" in experiment
        assert "name" in experiment
        assert "description" in experiment
        assert "data_sources" in experiment
        assert "status" in experiment
        assert "created_at" in experiment
        assert "config" in experiment
    
    def test_get_experiment_info(self, client):
        """Test getting specific experiment information."""
        response = client.get("/api/v1/experiments/exp_001")
        assert response.status_code == 200
        experiment = response.json()
        assert experiment["id"] == "exp_001"
        assert "High Content Screening" in experiment["name"]
    
    def test_get_nonexistent_experiment(self, client):
        """Test getting non-existent experiment."""
        response = client.get("/api/v1/experiments/nonexistent")
        assert response.status_code == 404
        assert "Experiment not found" in response.json()["detail"]

class TestDatasetEndpoints:
    """Test dataset management endpoints."""
    
    def test_list_datasets(self, client):
        """Test listing datasets."""
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        
        # Check dataset structure
        dataset = datasets[0]
        assert "name" in dataset
        assert "description" in dataset
        assert "data_type" in dataset
        assert "format" in dataset
        assert "size" in dataset
        assert "source" in dataset
        assert "last_updated" in dataset
    
    def test_get_dataset_info(self, client):
        """Test getting specific dataset information."""
        response = client.get("/api/v1/datasets/ChEMBL_compounds")
        assert response.status_code == 200
        dataset = response.json()
        assert dataset["name"] == "ChEMBL_compounds"
        assert "Chemical database" in dataset["description"]
    
    def test_get_nonexistent_dataset(self, client):
        """Test getting non-existent dataset."""
        response = client.get("/api/v1/datasets/nonexistent")
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

class TestConfigurationEndpoints:
    """Test configuration validation endpoints."""
    
    def test_validate_valid_config(self, client):
        """Test validating valid configuration."""
        response = client.post("/api/v1/validate-config", json=SAMPLE_CONFIG)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0
    
    def test_validate_invalid_config(self, client):
        """Test validating invalid configuration."""
        invalid_config = {"invalid_field": "value"}
        response = client.post("/api/v1/validate-config", json=invalid_config)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
    
    def test_validate_config_with_warnings(self, client):
        """Test validating configuration with warnings."""
        config_with_warnings = {
            "experiment_type": "test",
            "data_source": "test",
            "model": {
                "batch_size": 256,  # Large batch size
                "learning_rate": 0.1  # High learning rate
            },
            "max_epochs": 2000  # Very high epoch count
        }
        
        response = client.post("/api/v1/validate-config", json=config_with_warnings)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["warnings"]) > 0

class TestSystemEndpoints:
    """Test system information endpoints."""
    
    def test_get_system_info(self, client):
        """Test getting system information."""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "python_version" in data
        assert "pytorch_version" in data
        assert "platform" in data
        assert "cpu_count" in data
        assert "memory_available" in data
        assert "gpu_available" in data
        assert "dependencies" in data
        
        # Check dependencies structure
        dependencies = data["dependencies"]
        assert isinstance(dependencies, dict)
        assert "pandas" in dependencies
        assert "fastapi" in dependencies
        assert "omegaconf" in dependencies

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self, client):
        """Test a complete workflow from upload to analysis."""
        # 1. Upload data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("gene,expression\n")
            f.write("gene_A,1.2\n")
            f.write("gene_B,0.8\n")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                files = {'file': ('test.csv', f, 'text/csv')}
                data = {'data_type': 'genomics'}
                upload_response = client.post("/api/v1/data/upload", files=files, data=data)
            
            assert upload_response.status_code == 200
            file_info = upload_response.json()
            
            # 2. Start analysis
            analysis_request = {
                "experiment_type": "causal_discovery",
                "data_source": "genomics",
                "file_path": file_info["file_path"],
                "model_config": {"method": "pc", "alpha": 0.05}
            }
            
            analysis_response = client.post("/api/v1/analysis/start", json=analysis_request)
            assert analysis_response.status_code == 200
            job_info = analysis_response.json()
            
            # 3. Check analysis status
            status_response = client.get(f"/api/v1/analysis/status/{job_info['job_id']}")
            assert status_response.status_code == 200
            
        finally:
            Path(temp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
