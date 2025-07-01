"""
Loss functions for OpenPerturbation models.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from .causal_losses import CausalVAELoss
from .fusion_losses import MultiModalFusionLoss

# Import ContrastiveLoss from training losses for compatibility
from ..training.training_losses import ContrastiveLoss

__all__ = ['CausalVAELoss', 'MultiModalFusionLoss', 'ContrastiveLoss'] 