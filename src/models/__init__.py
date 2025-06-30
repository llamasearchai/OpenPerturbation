"""Model implementations for OpenPerturbations."""

from .vision.cell_vit import CellViT, CellularMorphologyEncoder
from .causal.causal_vae import CausalVAE, CausalDiscoveryModule
from .fusion.multimodal_transformer import MultiModalFusion

__all__ = [
    "CellViT",
    "CellularMorphologyEncoder",
    "CausalVAE",
    "CausalDiscoveryModule",
    "MultiModalFusion",
]

"""Models registry."""

import logging

logger = logging.getLogger(__name__)

# Initialize empty model registry
MODEL_REGISTRY = {}

def register_model(name: str, model_class):
    """Register a model in the registry."""
    MODEL_REGISTRY[name] = model_class
    logger.info(f"Registered model: {name}")

def get_model(name: str):
    """Get a model from the registry."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    else:
        logger.warning(f"Model {name} not found in registry")
        return None

def list_models():
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())

# Register some dummy models for demo
class DummyModel:
    """Dummy model for testing."""
    def __init__(self, *args, **kwargs):
        pass
        
    def forward(self, x):
        return x
        
    def predict(self, x):
        return x

# Register dummy models
register_model("multimodal_fusion", DummyModel)
register_model("causal_vae", DummyModel)
register_model("cell_vit", DummyModel)
register_model("molecular_gnn", DummyModel)
