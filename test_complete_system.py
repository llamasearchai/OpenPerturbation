#!/usr/bin/env python3
"""
Comprehensive System Test for OpenPerturbation

This script validates all the fixes implemented and ensures the entire system works perfectly.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def test_genomics_loader():
    """Test genomics data loader functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Genomics Data Loader...")
    
    try:
        from src.data.loaders.genomics_loader import GenomicsDataLoader, test_genomics_loader
        
        # Run the built-in test
        test_genomics_loader()
        
        # Additional validation
        config = {
            "data_dir": "test_genomics_data",
            "batch_size": 4,
            "normalize": True,
            "log_transform": True,
            "filter_genes": False,
            "n_cells": 50,
            "n_genes": 100
        }
        
        loader = GenomicsDataLoader(config)
        loader.setup()
        
        # Test dataset statistics
        stats = loader.get_dataset_statistics()
        assert 'train' in stats, "Train dataset not found"
        assert stats['train']['n_cells'] > 0, "No cells in train dataset"
        
        logger.info("✅ Genomics Data Loader: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Genomics Data Loader: FAILED - {e}")
        traceback.print_exc()
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Pipeline Initialization...")
    
    try:
        from src.pipeline.openperturbation_pipeline import OpenPerturbationPipeline
        from omegaconf import DictConfig
        
        config = DictConfig({
            'seed': 42,
            'use_gpu': False,
            'output_dir': 'test_outputs',
            'data': {
                'data_dir': 'test_genomics_data',
                'batch_size': 4
            },
            'model': {},
            'causal_discovery': {},
            'run_training': False,
            'run_causal_discovery': False,
            'run_explainability': False,
            'run_intervention_design': False
        })
        
        pipeline = OpenPerturbationPipeline(config)
        assert pipeline.config is not None, "Pipeline config not set"
        assert pipeline.output_dir.exists(), "Output directory not created"
        
        logger.info("✅ Pipeline Initialization: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline Initialization: FAILED - {e}")
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoints."""
    logger = logging.getLogger(__name__)
    logger.info("Testing API Endpoints...")
    
    try:
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get('/health')
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        health_data = response.json()
        assert health_data['status'] == 'healthy', "Health status not healthy"
        
        # Test models endpoint
        response = client.get('/models')
        assert response.status_code == 200, f"Models endpoint failed: {response.status_code}"
        models_data = response.json()
        assert len(models_data) > 0, "No models found"
        
        # Test system info endpoint
        response = client.get('/system/info')
        assert response.status_code == 200, f"System info failed: {response.status_code}"
        system_data = response.json()
        assert 'python_version' in system_data, "Python version not in system info"
        
        logger.info("✅ API Endpoints: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ API Endpoints: FAILED - {e}")
        traceback.print_exc()
        return False

def test_model_imports():
    """Test model imports."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Model Imports...")
    
    try:
        # Test vision models
        from src.models.vision.cell_vit import CellViT
        model = CellViT(config={'image_size': 224, 'patch_size': 16, 'num_classes': 10})
        assert model is not None, "CellViT model creation failed"
        
        # Test graph models
        from src.models.graph.molecular_gnn import MolecularGNN
        gnn = MolecularGNN(config={'node_features': 64, 'hidden_dim': 128, 'num_layers': 3})
        assert gnn is not None, "MolecularGNN model creation failed"
        
        # Test fusion models
        from src.models.fusion.multimodal_transformer import MultiModalFusion
        from omegaconf import DictConfig
        transformer = MultiModalFusion(DictConfig({
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6
        }))
        assert transformer is not None, "MultiModalFusion creation failed"
        
        logger.info("✅ Model Imports: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Imports: FAILED - {e}")
        traceback.print_exc()
        return False

def test_causal_discovery():
    """Test causal discovery functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Causal Discovery...")
    
    try:
        from src.causal.causal_discovery_engine import CausalDiscoveryEngine, run_causal_discovery
        import numpy as np
        
        # Create test data
        n_samples = 100
        n_variables = 10
        X = np.random.randn(n_samples, n_variables)
        perturbations = np.random.choice(['control', 'treated'], size=n_samples)
        
        # Test causal discovery
        config = {
            'method': 'pc',
            'alpha': 0.05,
            'max_conditioning_set_size': 3
        }
        
        results = run_causal_discovery(
            causal_factors=X,
            perturbation_labels=perturbations,
            config=config
        )
        
        assert 'adjacency_matrix' in results, "Adjacency matrix not in results"
        assert 'method' in results, "Method not in results"
        
        logger.info("✅ Causal Discovery: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Causal Discovery: FAILED - {e}")
        traceback.print_exc()
        return False

def test_explainability():
    """Test explainability functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Explainability...")
    
    try:
        from src.explainability.attention_maps import generate_attention_analysis
        from src.explainability.concept_activation import compute_concept_activations
        from src.explainability.pathway_analysis import run_pathway_analysis
        import torch
        
        # Test attention analysis with dummy model
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.attention = torch.nn.MultiheadAttention(64, 8)
                
            def forward(self, x):
                return self.conv(x)
        
        model = DummyModel()
        images = torch.randn(4, 3, 224, 224)
        perturbations = ['control', 'treated', 'control', 'treated']
        
        # This should not fail even if some features are disabled
        attention_results = generate_attention_analysis(
            model=model,
            images=images,
            perturbations=perturbations,
            output_dir="test_attention"
        )
        
        # Test pathway analysis
        gene_list = [f"GENE_{i}" for i in range(20)]
        pathway_results = run_pathway_analysis(
            gene_list=gene_list,
            output_dir="test_pathway"
        )
        
        logger.info("✅ Explainability: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Explainability: FAILED - {e}")
        traceback.print_exc()
        return False

def test_training_modules():
    """Test training modules."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Training Modules...")
    
    try:
        from src.training.lightning_modules import CausalVAELightningModule, MultiModalFusionModule
        from src.training.data_modules import PerturbationDataModule
        from omegaconf import DictConfig
        
        # Test lightning modules
        config = DictConfig({
            'model': {
                'latent_dim': 64,
                'input_dim': 1000,
                'causal_dim': 10
            },
            'loss': {
            },
            'training': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'batch_size': 4
            }
        })
        
        causal_module = CausalVAELightningModule(config)
        assert causal_module is not None, "CausalVAE module creation failed"
        
        fusion_config = DictConfig({
            'model': {
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4
            },
            'loss': {},
            'training': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'batch_size': 4
            }
        })
        fusion_module = MultiModalFusionModule(fusion_config)
        
        # Data module config
        data_config = DictConfig({
            'data_dir': 'test_genomics_data',
            'batch_size': 4,
            'num_workers': 0
        })
        data_module = PerturbationDataModule(data_config)
        assert data_module is not None, "Data module creation failed"
        
        logger.info("✅ Training Modules: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training Modules: FAILED - {e}")
        traceback.print_exc()
        return False

def test_agent_functionality():
    """Test agent functionality (without OpenAI)."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Agent Functionality...")
    
    try:
        from src.agents.agent_tools import AgentTools, PerturbationAnalysisTools
        from src.agents.conversation_handler import ConversationHandler
        
        # Test agent tools
        test_data = {'expression': [[1, 2, 3], [4, 5, 6]], 'perturbation': ['control', 'treated']}
        formatted = AgentTools.format_data_for_ai(test_data)
        assert isinstance(formatted, str), "Data formatting failed"
        
        # Test response parsing
        valid_json = '{"analysis": "test", "confidence": 0.8}'
        parsed = AgentTools.parse_ai_response(valid_json)
        assert parsed['analysis'] == 'test', "JSON parsing failed"
        
        # Test prompt building
        prompt = AgentTools.build_analysis_prompt(test_data, query="test analysis")
        assert isinstance(prompt, str), "Prompt building failed"
        assert "test analysis" in prompt, "Prompt content missing"
        
        # Test experiment validation
        valid_design = {
            'type': 'dose_response',
            'concentrations': [10.0],
            'timepoints': [24],
            'replicates': 3
        }
        validation_result = AgentTools.validate_experiment_design(valid_design)
        assert validation_result['valid'], f"Valid design rejected: {validation_result['errors']}"
        
        # Test conversation handler
        handler = ConversationHandler("test_user")
        handler.add_message("user", "test message")
        history = handler.get_conversation_history()
        assert len(history) == 1, "Conversation history not working"
        
        logger.info("✅ Agent Functionality: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent Functionality: FAILED - {e}")
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    logger = logging.getLogger(__name__)
    logger.info("Cleaning up test files...")
    
    import shutil
    
    test_dirs = [
        'test_genomics_data',
        'test_outputs',
        'test_attention',
        'test_pathway'
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            logger.info(f"Removed {test_dir}")

def main():
    """Run all tests."""
    logger = setup_logging()
    logger.info("🚀 Starting Comprehensive System Test for OpenPerturbation")
    logger.info("=" * 60)
    
    tests = [
        ("Genomics Data Loader", test_genomics_loader),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("API Endpoints", test_api_endpoints),
        ("Model Imports", test_model_imports),
        ("Causal Discovery", test_causal_discovery),
        ("Explainability", test_explainability),
        ("Training Modules", test_training_modules),
        ("Agent Functionality", test_agent_functionality),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED - Unexpected error: {e}")
            traceback.print_exc()
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"✅ Tests Passed: {passed}")
    logger.info(f"❌ Tests Failed: {failed}")
    logger.info(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! System is working perfectly!")
        cleanup_test_files()
        return 0
    else:
        logger.error(f"⚠️  {failed} tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 