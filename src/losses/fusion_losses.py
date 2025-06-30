"""
Fusion loss functions for multimodal models.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class MultiModalFusionLoss(nn.Module):
    """Loss function for multimodal fusion models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.prediction_weight = config.get('prediction_weight', 1.0)
        self.consistency_weight = config.get('consistency_weight', 0.5)
        self.diversity_weight = config.get('diversity_weight', 0.1)
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multimodal fusion losses."""
        
        losses = {}
        
        # Main prediction loss
        if 'predictions' in outputs and 'labels' in targets:
            losses['prediction'] = F.mse_loss(
                outputs['predictions'], targets['labels']
            )
        
        # Modality consistency loss
        if 'modality_features' in outputs:
            modality_features = outputs['modality_features']
            if len(modality_features) > 1:
                consistency_loss = torch.tensor(0.0, device=modality_features[0].device)
                for i in range(len(modality_features)):
                    for j in range(i + 1, len(modality_features)):
                        consistency_loss += F.mse_loss(
                            modality_features[i], modality_features[j]
                        )
                losses['consistency'] = consistency_loss / (len(modality_features) * (len(modality_features) - 1) / 2)
        
        # Feature diversity loss (encourage diverse representations)
        if 'features' in outputs:
            features = outputs['features']
            correlation_matrix = torch.corrcoef(features.T)
            # Penalize high correlations (encourage diversity)
            off_diagonal = correlation_matrix - torch.diag(torch.diag(correlation_matrix))
            losses['diversity'] = torch.mean(torch.abs(off_diagonal))
        
        # Compute total loss
        total_loss = torch.tensor(0.0, device=list(outputs.values())[0].device if outputs else torch.device('cpu'))
        
        for loss_name, loss_value in losses.items():
            if loss_name == 'prediction':
                total_loss += self.prediction_weight * loss_value
            elif loss_name == 'consistency':
                total_loss += self.consistency_weight * loss_value
            elif loss_name == 'diversity':
                total_loss += self.diversity_weight * loss_value
        
        losses['total'] = total_loss
        
        return losses 