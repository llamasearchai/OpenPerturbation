"""
Multimodal Transformer for perturbation biology data fusion.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from omegaconf import DictConfig


class MultiModalFusion(nn.Module):
    """Multimodal fusion model using transformer architecture."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dropout = config.get('dropout', 0.1)
        
        # Input projections for different modalities
        self.genomics_projection = nn.Linear(2000, self.hidden_dim)  # Assume 2000 genes
        self.imaging_projection = nn.Linear(2048, self.hidden_dim)   # Assume 2048 image features
        self.molecular_projection = nn.Linear(512, self.hidden_dim)  # Assume 512 molecular features
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output heads
        self.prediction_head = nn.Linear(self.hidden_dim, config.get('num_classes', 1))
        self.feature_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        
        # Extract features from different modalities
        features = []
        
        # Genomics features
        if 'genomics' in batch:
            genomics_features = batch['genomics']['expression']
            genomics_features = self.genomics_projection(genomics_features)
            features.append(genomics_features.unsqueeze(1))  # Add sequence dimension
        
        # Imaging features  
        if 'imaging' in batch:
            imaging_features = batch['imaging']['features']
            imaging_features = self.imaging_projection(imaging_features)
            features.append(imaging_features.unsqueeze(1))
        
        # Molecular features
        if 'molecular' in batch:
            molecular_features = batch['molecular']['features']
            molecular_features = self.molecular_projection(molecular_features)
            features.append(molecular_features.unsqueeze(1))
        
        # Concatenate all features
        if features:
            multimodal_features = torch.cat(features, dim=1)  # [batch_size, num_modalities, hidden_dim]
        else:
            # Fallback: create dummy features
            batch_size = batch.get('batch_size', 32)
            multimodal_features = torch.randn(batch_size, 3, self.hidden_dim, device=next(self.parameters()).device)
        
        # Apply transformer
        transformed_features = self.transformer(multimodal_features)
        
        # Pool features (mean pooling across modalities)
        pooled_features = torch.mean(transformed_features, dim=1)
        
        # Generate outputs
        outputs = {
            'predictions': self.prediction_head(pooled_features),
            'features': self.feature_head(pooled_features),
            'attention_weights': None  # Could add attention visualization
        }
        
        return outputs 