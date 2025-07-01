"""
Attention Map Analysis for Explainable AI in Perturbation Biology.

This module provides tools for generating and analyzing attention maps from
vision transformers and other attention-based models used in biological
image analysis and perturbation studies.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Import sklearn with fallback
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Clustering features disabled.")
    
    class DummyKMeans:
        def __init__(self, n_clusters=8):
            self.n_clusters = n_clusters
        def fit(self, X):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    
    KMeans = DummyKMeans

# Image processing and visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except Exception:  # Catch all exceptions during matplotlib import
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features disabled.")
    
    # Create dummy matplotlib-like functionality
    class DummyPlt:
        @staticmethod
        def figure(*args, **kwargs):
            return None
        @staticmethod
        def subplot(*args, **kwargs):
            return None
        @staticmethod
        def subplots(*args, **kwargs):
            return None, None
        @staticmethod
        def imshow(*args, **kwargs):
            return None
        @staticmethod
        def colorbar(*args, **kwargs):
            return None
        @staticmethod
        def title(*args, **kwargs):
            return None
        @staticmethod
        def savefig(*args, **kwargs):
            pass
        @staticmethod
        def show(*args, **kwargs):
            pass
        @staticmethod
        def close(*args, **kwargs):
            pass
        @staticmethod
        def tight_layout(*args, **kwargs):
            pass
    
    plt = DummyPlt()
    sns = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:  # Catch all exceptions during OpenCV import
    OPENCV_AVAILABLE = False
    warnings.warn("OpenCV not available. Some image processing features disabled.")
    
    # Create dummy cv2 functionality
    class DummyCV2:
        @staticmethod
        def resize(img, size, *args, **kwargs):
            return img
        @staticmethod
        def GaussianBlur(img, *args, **kwargs):
            return img
        @staticmethod
        def applyColorMap(img, *args, **kwargs):
            return img
        COLORMAP_JET = 0
    
    cv2 = DummyCV2()

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except Exception:  # Catch all exceptions during SciPy import
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some filtering features disabled.")
    
    # Create dummy scipy functionality
    def gaussian_filter(img, sigma, *args, **kwargs):
        return img

try:
    from skimage import filters, transform
    SKIMAGE_AVAILABLE = True
except Exception:  # Catch all exceptions during skimage import
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available. Some image processing features disabled.")
    
    # Create dummy skimage functionality
    class DummyFilters:
        @staticmethod
        def gaussian(img, sigma, *args, **kwargs):
            return img
    
    class DummyTransform:
        @staticmethod
        def resize(img, output_shape, *args, **kwargs):
            return img
    
    filters = DummyFilters()
    transform = DummyTransform()

logger = logging.getLogger(__name__)


class AttentionMapExtractor:
    """Extract attention maps from transformer models."""

    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        self.model = model
        self.layer_names = layer_names or self._get_default_layer_names()
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_attention_hooks()

    def _get_default_layer_names(self) -> List[str]:
        """Get default attention layer names from the model."""
        layer_names = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                layer_names.append(name)
        return layer_names[:4]  # Limit to first 4 layers

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention maps."""

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attention_weights = output[1] if len(output) > 1 else output[0]
                else:
                    attention_weights = output

                if isinstance(attention_weights, torch.Tensor):
                    self.attention_maps[name] = attention_weights.detach().cpu()

            return hook

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)

    def extract_attention_maps(
        self, images: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps from input images."""
        self.attention_maps.clear()

        with torch.no_grad():
            outputs = self.model(images)

        if return_features:
            return {"attention_maps": self.attention_maps.copy(), "features": outputs}

        return self.attention_maps.copy()

    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.cleanup_hooks()


class AttentionRollout:
    """Compute attention rollout for better visualization."""

    def __init__(self, discard_ratio: float = 0.9):
        self.discard_ratio = discard_ratio

    def compute_rollout(self, attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention rollout across layers."""
        if not attention_weights:
            return torch.zeros((1, 1))

        # Start with identity matrix
        result = torch.eye(attention_weights[0].size(-1))

        for attention in attention_weights:
            # Average across heads if multi-head
            if attention.dim() == 4:  # [batch, heads, seq, seq]
                attention = attention.mean(dim=1)

            # Apply discard ratio
            attention = self.apply_threshold(attention[0])  # Take first batch

            # Add residual connection
            attention = attention + torch.eye(attention.size(-1))
            attention = attention / attention.sum(dim=-1, keepdim=True)

            # Multiply with previous result
            result = torch.matmul(attention, result)

        return result

    def apply_threshold(self, attention: torch.Tensor) -> torch.Tensor:
        """Apply threshold to attention weights."""
        flat_attention = attention.view(-1)
        threshold_idx = int(len(flat_attention) * self.discard_ratio)
        threshold_val = torch.sort(flat_attention)[0][threshold_idx]

        attention = torch.where(attention < threshold_val, torch.zeros_like(attention), attention)

        return attention


class BiologicalAttentionAnalyzer:
    """Analyze attention maps in the context of biological structures."""

    def __init__(
        self,
        cellular_regions: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        organelle_masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.cellular_regions = cellular_regions or self._get_default_regions()
        self.organelle_masks = organelle_masks or {}

    def _get_default_regions(self) -> Dict[str, List[Tuple[int, int]]]:
        """Define default cellular regions for analysis."""
        return {
            "nucleus": [(112, 112)],  # Center region
            "cytoplasm": [(56, 56), (168, 56), (56, 168), (168, 168)],  # Corners
            "membrane": [(0, 112), (224, 112), (112, 0), (112, 224)],  # Edges
            "mitochondria": [(84, 84), (140, 140)],  # Scattered regions
        }

    def analyze_regional_attention(
        self, attention_maps: torch.Tensor, patch_size: int = 16, image_size: int = 224
    ) -> Dict[str, float]:
        """Analyze attention distribution across cellular regions."""
        if attention_maps.dim() == 4:
            attention_maps = attention_maps.mean(dim=1)  # Average across heads

        if attention_maps.dim() == 3:
            attention_maps = attention_maps[0]  # Take first batch

        # Convert to spatial attention map
        seq_len = attention_maps.size(-1)
        grid_size = int(np.sqrt(seq_len - 1))  # Exclude CLS token

        # Remove CLS token and reshape
        spatial_attention = attention_maps[0, 1:].view(grid_size, grid_size)

        # Upsample to image resolution
        spatial_attention = torch.nn.functional.interpolate(
            spatial_attention.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        regional_scores = {}
        for region_name, coordinates in self.cellular_regions.items():
            total_attention = 0.0
            for x, y in coordinates:
                # Define region around coordinate
                x_start = max(0, x - patch_size // 2)
                x_end = min(image_size, x + patch_size // 2)
                y_start = max(0, y - patch_size // 2)
                y_end = min(image_size, y + patch_size // 2)

                region_attention = spatial_attention[y_start:y_end, x_start:x_end].mean()
                total_attention += region_attention.item()

            regional_scores[region_name] = total_attention / len(coordinates)

        return regional_scores


class AttentionVisualization:
    """Create visualizations of attention maps."""

    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize

    def plot_attention_overlay(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        title: str = "Attention Overlay",
        alpha: float = 0.6,
        colormap: str = "jet",
    ):
        """Create attention overlay visualization."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)

        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Attention map
        im2 = ax2.imshow(attention_map, cmap=colormap)
        ax2.set_title("Attention Map")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2)

        # Overlay
        ax3.imshow(image)
        ax3.imshow(attention_map, cmap=colormap, alpha=alpha)
        ax3.set_title(title)
        ax3.axis("off")

        plt.tight_layout()
        return fig


# Main analysis function
def generate_attention_analysis(
    model: nn.Module,
    images: torch.Tensor,
    perturbations: torch.Tensor,
    output_dir: str = "attention_analysis",
) -> Dict[str, Any]:
    """Generate comprehensive attention analysis for perturbation biology."""
    logger.info("Starting attention analysis...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    extractor = AttentionMapExtractor(model)
    rollout = AttentionRollout()
    bio_analyzer = BiologicalAttentionAnalyzer()
    visualizer = AttentionVisualization()

    analysis_results = {
        "attention_maps": {},
        "rollout_maps": {},
        "regional_analysis": {},
        "statistics": {},
        "visualizations": [],
    }

    for i, (image, perturbation) in enumerate(zip(images, perturbations)):
        logger.info(f"Processing sample {i+1}/{len(images)}")

        # Extract attention maps
        attention_maps = extractor.extract_attention_maps(image.unsqueeze(0))

        # Compute rollout
        attention_list = list(attention_maps.values())
        rollout_map = rollout.compute_rollout(attention_list)

        # Regional analysis
        if attention_list:
            regional_scores = bio_analyzer.analyze_regional_attention(attention_list[0])
        else:
            regional_scores = {}

        # Store results
        sample_key = f"sample_{i}"
        analysis_results["attention_maps"][sample_key] = {
            name: att.numpy() for name, att in attention_maps.items()
        }
        analysis_results["rollout_maps"][sample_key] = rollout_map.numpy()
        analysis_results["regional_analysis"][sample_key] = regional_scores

        # Create visualizations
        if i < 5 and attention_list:  # Limit visualizations
            image_np = image.permute(1, 2, 0).numpy()

            # Attention overlay
            spatial_attention = attention_list[0][0, 0, 1:]
            grid_size = int(np.sqrt(len(spatial_attention)))
            spatial_attention = spatial_attention.view(grid_size, grid_size).numpy()

            fig = visualizer.plot_attention_overlay(
                image_np, spatial_attention, title=f"Sample {i+1} Attention"
            )

            viz_path = output_path / f"attention_overlay_{i}.png"
            fig.savefig(viz_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            analysis_results["visualizations"].append(str(viz_path))

    # Compute summary statistics
    all_regional_scores = analysis_results["regional_analysis"]
    if all_regional_scores:
        first_scores = next(iter(all_regional_scores.values()))
        if first_scores:
            region_names = list(first_scores.keys())

            statistics = {}
            for region in region_names:
                region_scores = [scores.get(region, 0.0) for scores in all_regional_scores.values()]
                statistics[region] = {
                    "mean": float(np.mean(region_scores)),
                    "std": float(np.std(region_scores)),
                    "min": float(np.min(region_scores)),
                    "max": float(np.max(region_scores)),
                }

            analysis_results["statistics"] = statistics

    # Export results
    results_file = output_path / "attention_analysis_results.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = analysis_results.copy()
        for sample_key in json_results["attention_maps"]:
            for layer_name in json_results["attention_maps"][sample_key]:
                json_results["attention_maps"][sample_key][layer_name] = json_results[
                    "attention_maps"
                ][sample_key][layer_name].tolist()
        for sample_key in json_results["rollout_maps"]:
            json_results["rollout_maps"][sample_key] = json_results["rollout_maps"][
                sample_key
            ].tolist()

        json.dump(json_results, f, indent=2)

    # Cleanup
    extractor.cleanup_hooks()

    logger.info(f"Attention analysis completed. Results saved to {output_path}")
    return analysis_results


# Export classes and functions that should be available in __init__.py
AttentionVisualizer = AttentionVisualization
AttentionAnalyzer = BiologicalAttentionAnalyzer


def create_attention_dashboard(
    results: Dict[str, Any], output_file: str = "attention_dashboard.html"
):
    """Create an interactive dashboard for attention analysis results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attention Analysis Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Attention Analysis Dashboard</h1>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div id="statistics">
                {_format_statistics_html(results.get('statistics', {}))}
            </div>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <div id="visualizations">
                {_format_visualizations_html(results.get('visualizations', []))}
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)

    logger.info(f"Dashboard created: {output_file}")


def _format_statistics_html(statistics: Dict[str, Dict[str, float]]) -> str:
    """Format statistics for HTML display."""
    html = ""
    for region, stats in statistics.items():
        html += f"""
        <div class="metric">
            <h3>{region.title()}</h3>
            <p>Mean: {stats.get('mean', 0):.3f}</p>
            <p>Std: {stats.get('std', 0):.3f}</p>
        </div>
        """
    return html


def _format_visualizations_html(visualizations: List[str]) -> str:
    """Format visualizations for HTML display."""
    html = ""
    for viz_path in visualizations:
        html += f'<img src="{viz_path}" style="max-width: 400px; margin: 10px;"/>'
    return html
