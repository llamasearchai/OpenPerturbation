#!/usr/bin/env python3
"""
OpenPerturbation Demo Script

Demonstrates the complete functionality of the OpenPerturbation platform.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
import logging
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pathlib import Path

def test_imports():
    """Test that all major components can be imported."""
    try:
        from src.training.data_modules import PerturbationDataModule
        print("SUCCESS: Data modules imported successfully")
        
        from src.causal.causal_discovery_engine import CausalDiscoveryEngine
        print("SUCCESS: Causal discovery imported successfully")
        
    except ImportError as e:
        print(f" Failed to import modules: {e}")
        return False
    return True

def demo_run(config_path: str = "configs/main_config.yaml"):
    """
    Runs a demonstration of the OpenPerturbation pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting OpenPerturbation Demo")
    
    # 1. Load Configuration
    # This part would typically use Hydra to load a YAML config
    # For this demo, we'll simulate a simple config object
    config = {
        "data": {"datamodule_name": "PerturbationDataModule"},
        "model": {"model_name": "CausalVAE"},
        "training": {"trainer": {"max_epochs": 1}},
        "pipeline": {
            "run_causal_discovery": True,
            "run_intervention_analysis": True,
            "run_explainability": False
        },
        "causal_discovery": {
            "discovery_method": "pc",
            "significance_level": 0.05
        }
    }
    
    logger.info("Configuration loaded.")

    # 2. Initialize Pipeline
    # In a real scenario, the pipeline would be initialized with the config
    # from src.pipeline.openperturbation_pipeline import OpenPerturbationPipeline
    # pipeline = OpenPerturbationPipeline(config)
    
    # For the demo, we'll manually instantiate components
    logger.info("Initializing pipeline components...")
    
    # Data Module
    try:
        from src.training.data_modules import PerturbationDataModule
        datamodule_config = DictConfig(config['data'])
        data_module = PerturbationDataModule(config=datamodule_config)
        data_module.setup()
        logger.info("Data module initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize data module: {e}")
        return

    # Causal Discovery Engine
    try:
        from src.causal.causal_discovery_engine import CausalDiscoveryEngine
        causal_engine = CausalDiscoveryEngine(config['causal_discovery'])
        
        # Create some dummy data for discovery
        dummy_data = np.random.rand(100, 10)
        
        results = causal_engine.discover_causal_structure(dummy_data)
        logger.info("Causal discovery finished.")
        logger.info(f"Discovered {results['analysis']['num_edges']} causal edges.")
        
    except Exception as e:
        logger.error(f"Failed to run causal discovery: {e}")

    logger.info("Demo Finished Successfully!")

if __name__ == "__main__":
    if test_imports():
        demo_run()
    else:
        sys.exit(1) 