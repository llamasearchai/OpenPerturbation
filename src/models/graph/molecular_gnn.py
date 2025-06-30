"""
Molecular Graph Neural Network for chemical compound analysis.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from omegaconf import DictConfig


class MolecularGNN(nn.Module):
    """Graph Neural Network for molecular property prediction."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        
        # Node and edge feature dimensions
        self.node_input_dim = config.get('node_input_dim', 75)  # Atom features
        self.edge_input_dim = config.get('edge_input_dim', 12)  # Bond features
        
        # Node and edge embeddings
        self.node_embedding = nn.Linear(self.node_input_dim, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.edge_input_dim, self.hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(GraphConvLayer(self.hidden_dim, self.dropout))
        
        # Graph pooling
        self.global_pool = GlobalAttentionPooling(self.hidden_dim)
        
        # Output heads
        self.property_head = nn.Linear(self.hidden_dim, config.get('num_properties', 1))
        self.feature_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        
        # Extract graph data
        if 'molecular' in batch and 'graph' in batch['molecular']:
            node_features = batch['molecular']['graph']['node_features']
            edge_features = batch['molecular']['graph']['edge_features']
            edge_index = batch['molecular']['graph']['edge_index']
            batch_index = batch['molecular']['graph']['batch']
        else:
            # Create dummy graph data
            batch_size = batch.get('batch_size', 32)
            num_nodes = 20
            node_features = torch.randn(batch_size * num_nodes, self.node_input_dim)
            edge_features = torch.randn(batch_size * num_nodes * 2, self.edge_input_dim)
            edge_index = torch.randint(0, batch_size * num_nodes, (2, batch_size * num_nodes * 2))
            batch_index = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
        
        # Embed features
        node_embeddings = self.node_embedding(node_features)
        edge_embeddings = self.edge_embedding(edge_features)
        
        # Apply graph convolutions
        for conv_layer in self.conv_layers:
            node_embeddings = conv_layer(node_embeddings, edge_embeddings, edge_index)
        
        # Global pooling
        graph_features = self.global_pool(node_embeddings, batch_index)
        
        # Generate outputs
        outputs = {
            'predictions': self.property_head(graph_features),
            'features': self.feature_head(graph_features),
            'node_embeddings': node_embeddings
        }
        
        return outputs


class GraphConvLayer(nn.Module):
    """Graph convolution layer."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
        # Message passing
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Create messages
        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        
        messages = torch.cat([source_features, target_features, edge_features], dim=-1)
        messages = self.message_net(messages)
        
        # Aggregate messages
        aggregated = torch.zeros_like(node_features)
        aggregated.scatter_add_(0, target_nodes.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        
        # Update node features
        updated_features = torch.cat([node_features, aggregated], dim=-1)
        updated_features = self.update_net(updated_features)
        
        # Residual connection and normalization
        output = self.norm(node_features + updated_features)
        
        return output


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling for graph-level representations."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
        # Compute attention weights
        attention_weights = self.attention_net(node_features)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Weighted sum by batch
        batch_size = batch_index.max().item() + 1
        graph_features = torch.zeros(batch_size, node_features.size(-1), device=node_features.device)
        
        for i in range(batch_size):
            mask = batch_index == i
            if mask.sum() > 0:
                batch_nodes = node_features[mask]
                batch_weights = attention_weights[mask]
                graph_features[i] = torch.sum(batch_nodes * batch_weights, dim=0)
        
        return graph_features 