"""
Loss functions for OpenPerturbation models.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from .causal_losses import CausalVAELoss
from .fusion_losses import MultiModalFusionLoss

__all__ = ['CausalVAELoss', 'MultiModalFusionLoss'] 