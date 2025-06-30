"""
Causal loss functions for VAE and causal discovery models.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class CausalVAELoss(nn.Module):
    """Loss function for Causal VAE."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.kl_weight = config.get('kl_weight', 1.0)
        self.causal_weight = config.get('causal_weight', 1.0)
        self.intervention_weight = config.get('intervention_weight', 1.0)
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute causal VAE losses."""
        
        losses = {}
        
        # Reconstruction loss
        if 'reconstruction' in outputs and 'input' in targets:
            losses['reconstruction'] = F.mse_loss(
                outputs['reconstruction'], targets['input']
            )
        
        # KL divergence loss
        if 'mu' in outputs and 'log_var' in outputs:
            losses['kl_divergence'] = -0.5 * torch.mean(
                1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp()
            )
        
        # Causal consistency loss
        if 'causal_factors' in outputs and 'causal_targets' in targets:
            losses['causal_consistency'] = F.mse_loss(
                outputs['causal_factors'], targets['causal_targets']
            )
        
        # Intervention loss
        if 'intervention_effect' in outputs and 'intervention_targets' in targets:
            losses['intervention'] = F.mse_loss(
                outputs['intervention_effect'], targets['intervention_targets']
            )
        
        # Compute total loss
        total_loss = torch.tensor(0.0, device=outputs['reconstruction'].device if 'reconstruction' in outputs else torch.device('cpu'))
        
        for loss_name, loss_value in losses.items():
            if loss_name == 'reconstruction':
                total_loss += self.reconstruction_weight * loss_value
            elif loss_name == 'kl_divergence':
                total_loss += self.kl_weight * loss_value
            elif loss_name == 'causal_consistency':
                total_loss += self.causal_weight * loss_value
            elif loss_name == 'intervention':
                total_loss += self.intervention_weight * loss_value
        
        losses['total'] = total_loss
        
        return losses 