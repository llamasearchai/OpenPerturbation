"""
Datasets module for OpenPerturbation

This module provides access to various datasets for perturbation biology research,
including integration with HuggingFace datasets and other public repositories.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

try:
    from .huggingface_integration import (
        HuggingFaceDatasetLoader,
        download_real_dataset_info
    )
    __all__ = [
        "HuggingFaceDatasetLoader",
        "download_real_dataset_info"
    ]
except ImportError:
    # Graceful fallback if dependencies are not available
    __all__ = []

# Version information
__version__ = "1.0.0" 