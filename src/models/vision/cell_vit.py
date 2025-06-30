"""Vision Transformer adapted for cellular imaging analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for cellular compartments."""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.pos_bias = nn.Parameter(torch.zeros(196, 196))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if N <= self.pos_bias.size(0):
            attn = attn + self.pos_bias[:N, :N].unsqueeze(0).unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out


class CellularChannelAttention(nn.Module):
    """Channel attention for different fluorescent markers."""

    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.num_channels = num_channels
        reduced_dim = max(1, num_channels // reduction_ratio)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_channels, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        channel_weights = self.global_pool(x).view(B, C)
        channel_weights = self.channel_mlp(channel_weights).view(B, C, 1, 1)

        return x * channel_weights


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class CellViT(nn.Module):
    """Vision Transformer optimized for cellular imaging analysis."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Base configuration
        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 16)
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)

        # Input channels
        self.num_input_channels = config.get("num_input_channels", 5)

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            self.num_input_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # Channel attention
        if config.get("channel_attention", True):
            self.channel_attention = CellularChannelAttention(self.num_input_channels)
        else:
            self.channel_attention = None

        # Enhanced spatial attention
        if config.get("use_spatial_attention", True):
            self.spatial_attention = SpatialAttention(
                self.hidden_size, config.get("spatial_attention_heads", 8)
            )
        else:
            self.spatial_attention = None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden_size, self.num_heads, config.get("drop_rate", 0.1))
                for _ in range(self.num_layers)
            ]
        )

        # Final layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Cellular feature heads
        self.morphology_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.localization_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CellViT."""
        batch_size = x.size(0)

        # Channel attention for fluorescent markers
        if self.channel_attention is not None:
            x = self.channel_attention(x)

        # Patch embedding
        x = self.patch_embedding(x)  # [B, hidden_size, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]

        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embedding

        # Transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x = block(x)
            if return_attention:
                # Store attention weights for analysis
                attention_weights.append(x)

        # Layer norm
        x = self.layer_norm(x)

        # Extract features
        cls_token = x[:, 0]  # [B, hidden_size]
        patch_features = x[:, 1:]  # [B, num_patches, hidden_size]

        # Enhanced spatial attention
        spatial_attn = None
        if self.spatial_attention is not None:
            patch_features, spatial_attn = self.spatial_attention(patch_features)
            # Update cls token with attended features
            cls_token = patch_features.mean(dim=1)

        # Cellular-specific features
        morphology_features = self.morphology_head(cls_token)
        localization_features = self.localization_head(cls_token)

        outputs = {
            "embeddings": cls_token,
            "patch_features": patch_features,
            "morphology_features": morphology_features,
            "localization_features": localization_features,
        }

        if return_attention:
            outputs["attention_weights"] = attention_weights
            if spatial_attn is not None:
                outputs["spatial_attention"] = spatial_attn

        return outputs

    def extract_cellular_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive cellular features."""
        outputs = self.forward(x, return_attention=True, return_features=False)

        cellular_features = {
            "morphology": outputs["morphology_features"],
            "localization": outputs["localization_features"],
            "global_context": outputs["embeddings"],
            "spatial_patterns": outputs["patch_features"],
        }

        if "attention_weights" in outputs:
            cellular_features["attention_patterns"] = outputs["attention_weights"]

        return cellular_features


class CellularMorphologyEncoder(nn.Module):
    """Specialized encoder for cellular morphology analysis."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Base vision model
        self.vision_backbone = CellViT(config)

        # Morphology-specific processing
        self.morphology_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Cell cycle phase classifier
        self.cell_cycle_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),  # G1, S, G2, M, apoptosis
        )

        # Cell health scorer
        self.health_scorer = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract morphology features and predictions."""
        # Extract visual features
        vision_outputs = self.vision_backbone(x)

        # Process morphology
        morphology_emb = self.morphology_projector(vision_outputs["embeddings"])

        # Make predictions
        cell_cycle_logits = self.cell_cycle_head(morphology_emb)
        health_score = self.health_scorer(morphology_emb)

        return {
            "morphology_embedding": morphology_emb,
            "cell_cycle_logits": cell_cycle_logits,
            "health_score": health_score,
            "raw_features": vision_outputs,
        }
