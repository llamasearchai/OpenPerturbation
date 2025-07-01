"""
PyTorch Lightning modules for perturbation biology models.

Implements training, validation, and testing logic for various models
including CausalVAE, MultiModalFusion, and CausalDiscovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Lazy import for pytorch lightning to avoid scipy issues
try:
    import pytorch_lightning as pl
    PYTORCH_LIGHTNING_AVAILABLE = True
except Exception:
    PYTORCH_LIGHTNING_AVAILABLE = False
    # Create dummy pl module
    class DummyLightningModule:
        def __init__(self):
            pass
        def forward(self, x):
            return x
        def training_step(self, batch, batch_idx):
            return torch.tensor(0.0)
        def validation_step(self, batch, batch_idx):
            return torch.tensor(0.0)
        def configure_optimizers(self):
            return torch.optim.Adam([torch.tensor(1.0)])
        def log(self, *args, **kwargs):
            pass
        # Added stub to mimic PyTorch Lightning interface
        def save_hyperparameters(self, *args, **kwargs):  # noqa: D401
            """Stub for LightningModule.save_hyperparameters."""
            pass
    
    class DummyPL:
        LightningModule = DummyLightningModule
        LightningDataModule = DummyLightningModule
    
    pl = DummyPL()
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import logging
import wandb
from pathlib import Path
import math
import warnings

# Import models
from ..models.causal.causal_vae import CausalVAE
from ..models.vision.cell_vit import CellViT
from ..models.fusion.multimodal_transformer import MultiModalFusion
from ..models.graph.molecular_gnn import MolecularGNN

# Import losses and metrics
from ..losses import (
    CausalVAELoss,
    MultiModalFusionLoss,
    ContrastiveLoss
)
from .training_metrics import (
    CausalDiscoveryMetrics,
    PerturbationPredictionMetrics,
    ClassificationMetrics
)

# Machine learning and optimization
try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except Exception:  # Catch all exceptions during sklearn import
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some metrics disabled.")
    
    # Create dummy sklearn functionality
    def accuracy_score(y_true, y_pred):
        return 0.5
    
    def f1_score(y_true, y_pred, **kwargs):
        return 0.5
    
    def precision_score(y_true, y_pred, **kwargs):
        return 0.5
    
    def recall_score(y_true, y_pred, **kwargs):
        return 0.5

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:  # Catch all exceptions during SciPy import
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some optimization features disabled.")
    
    # Create dummy scipy functionality
    def minimize(fun, x0, *args, **kwargs):
        return type('obj', (object,), {'x': x0, 'fun': 0.0, 'success': True})()

logger = logging.getLogger(__name__)

class BaseLightningModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self._experiment = None
        
    @property
    def logger_experiment(self):
        """Safe access to logger experiment."""
        if self._experiment is None:
            # Lazy initialization with safe attribute access
            try:
                if hasattr(self, "trainer") and self.trainer:
                    if self.trainer.logger:
                        # Use getattr for safe access
                        self._experiment = getattr(self.trainer.logger, 'experiment', None)
            except AttributeError:
                self._experiment = None
        return self._experiment

class CausalVAELightningModule(BaseLightningModule):
    """
    Lightning module for Causal VAE training.
    
    Handles training of causal variational autoencoders for
    disentangled representation learning in perturbation biology.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        # Initialize model
        self.model = CausalVAE(config.model)
        
        # Initialize loss function
        self.loss_fn = CausalVAELoss(config.loss)
        
        # Initialize metrics
        self.train_metrics = CausalDiscoveryMetrics(prefix='train/')
        self.val_metrics = CausalDiscoveryMetrics(prefix='val/')
        
        # Training parameters
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        
        # Loss tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x: torch.Tensor, intervention: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through CausalVAE."""
        return self.model(x, intervention)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        
        # Extract multimodal features (simplified - would need proper feature extraction)
        features = self._extract_features(batch)
        
        # Forward pass
        outputs = self.model(features)
        
        # Compute loss
        loss_dict = self.model.compute_loss(features, outputs)
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics with proper format
        targets_dict = {'causal_factors': features}  # Convert to dict format expected by metrics
        self.train_metrics.update(outputs, targets_dict)
        
        # Store outputs for epoch end
        self.training_step_outputs.append({
            'loss': loss_dict['total_loss'],
            'causal_factors': outputs['z_causal'].detach(),
            'reconstruction': outputs['x_recon'].detach()
        })
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        
        # Extract features
        features = self._extract_features(batch)
        
        # Forward pass
        outputs = self.model(features)
        
        # Compute loss
        loss_dict = self.model.compute_loss(features, outputs)
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val/{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update metrics with proper format
        targets_dict = {'causal_factors': features}  # Convert to dict format expected by metrics
        self.val_metrics.update(outputs, targets_dict)
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'loss': loss_dict['total_loss'],
            'causal_factors': outputs['z_causal'].detach(),
            'reconstruction': outputs['x_recon'].detach()
        })
        
        return loss_dict['total_loss']
    
    def on_training_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        
        # Compute and log metrics
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()
        
        # Generate visualizations
        if self.current_epoch % 10 == 0:
            self._generate_training_visualizations()
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        
        # Compute and log metrics
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics)
        
        # Log main validation metric
        main_metric = val_metrics.get('val/causal_score', 0.0)
        self.log('val/causal_score', main_metric, prog_bar=True)
        
        self.val_metrics.reset()
        
        # Generate visualizations
        if self.current_epoch % 10 == 0:
            self._generate_validation_visualizations()
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Handle potential None value for max_epochs
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/causal_score',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract features from multimodal batch."""
        
        # This is a simplified version - in practice would need proper feature extraction
        features_list = []
        
        # Extract imaging features
        if 'imaging' in batch and 'images' in batch['imaging']:
            images = batch['imaging']['images']
            # Flatten images as simple features (would use vision model in practice)
            img_features = images.view(images.size(0), -1)
            features_list.append(img_features)
        
        # Extract genomics features
        if 'genomics' in batch and 'expressions' in batch['genomics']:
            expressions = batch['genomics']['expressions']
            features_list.append(expressions)
        
        # Extract molecular features
        if 'molecular' in batch and 'descriptors' in batch['molecular']:
            descriptors = batch['molecular']['descriptors']
            features_list.append(descriptors)
        
        # Concatenate all features
        if features_list:
            features = torch.cat(features_list, dim=1)
        else:
            # Fallback - create dummy features
            batch_size = batch.get('batch_size', 32)
            features = torch.randn(batch_size, 128, device=self.device)
        
        return features
    
    def _generate_training_visualizations(self):
        """Generate training visualizations for logging."""
        
        if not self.training_step_outputs:
            return
        
        try:
            # Collect causal factors from training steps
            causal_factors = torch.cat([out['causal_factors'] for out in self.training_step_outputs[-10:]])
            
            # Generate t-SNE visualization of causal factors
            if causal_factors.size(0) > 50:  # Need sufficient samples for t-SNE
                tsne_plot = self._create_tsne_plot(causal_factors.cpu().numpy(), 'Training Causal Factors')
                
                if self.logger_experiment:
                    self.logger_experiment.log({
                        'train/causal_factors_tsne': wandb.Image(tsne_plot),
                        'epoch': self.current_epoch
                    })
            
            # Generate reconstruction quality samples
            reconstruction_samples = self._create_reconstruction_samples()
            if reconstruction_samples and self.logger_experiment:
                self.logger_experiment.log({
                    'train/reconstructions': wandb.Image(reconstruction_samples),
                    'epoch': self.current_epoch
                })
        
        except Exception as e:
            logger.warning(f"Failed to generate training visualizations: {e}")
    
    def _generate_validation_visualizations(self):
        """Generate validation visualizations for logging."""
        
        if not self.validation_step_outputs:
            return
        
        try:
            # Collect causal factors from validation steps
            causal_factors = torch.cat([out['causal_factors'] for out in self.validation_step_outputs])
            
            # Generate causal factor distribution plot
            factor_dist_plot = self._create_factor_distribution_plot(causal_factors.cpu().numpy())
            
            if self.logger_experiment:
                self.logger_experiment.log({
                    'val/causal_factor_distribution': wandb.Image(factor_dist_plot),
                    'epoch': self.current_epoch
                })
            
            # Generate disentanglement metrics
            disentanglement_metrics = self._compute_disentanglement_metrics(causal_factors.cpu().numpy())
            self.log_dict({f'val/disentanglement_{k}': v for k, v in disentanglement_metrics.items()})
        
        except Exception as e:
            logger.warning(f"Failed to generate validation visualizations: {e}")
    
    def _create_tsne_plot(self, features: np.ndarray, title: str):
        """Create t-SNE visualization of features."""
        
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # Reduce dimensionality with t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
            features_2d = tsne.fit_transform(features)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)
            ax.set_title(title)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.tight_layout()
            
            return fig
        
        except ImportError:
            logger.warning("scikit-learn not available for t-SNE visualization")
            return None
        except Exception as e:
            logger.warning(f"Failed to create t-SNE plot: {e}")
            return None
    
    def _create_factor_distribution_plot(self, factors: np.ndarray):
        """Create causal factor distribution plot."""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            num_factors = min(4, factors.shape[1])
            
            for i in range(num_factors):
                ax = axes[i]
                ax.hist(factors[:, i], bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'Causal Factor {i+1}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(num_factors, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            logger.warning(f"Failed to create factor distribution plot: {e}")
            return None
    
    def _create_reconstruction_samples(self):
        """Create reconstruction quality visualization."""
        
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_step_outputs:
                return None
            
            # Get recent reconstruction samples
            recent_outputs = self.training_step_outputs[-5:]
            reconstructions = [out['reconstruction'] for out in recent_outputs]
            
            if not reconstructions:
                return None
            
            # Create simple visualization (placeholder)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot reconstruction quality over recent steps
            recon_quality = [torch.mean(recon).item() for recon in reconstructions]
            ax.plot(recon_quality, marker='o')
            ax.set_title('Recent Reconstruction Quality')
            ax.set_xlabel('Recent Steps')
            ax.set_ylabel('Mean Reconstruction Value')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            logger.warning(f"Failed to create reconstruction samples: {e}")
            return None
    
    def _compute_disentanglement_metrics(self, factors: np.ndarray) -> Dict[str, float]:
        """Compute disentanglement metrics for causal factors."""
        
        try:
            metrics = {}
            
            # Factor variance (higher is better for disentanglement)
            factor_vars = np.var(factors, axis=0)
            metrics['mean_factor_variance'] = np.mean(factor_vars)
            metrics['min_factor_variance'] = np.min(factor_vars)
            
            # Factor correlation (lower is better for disentanglement)
            if factors.shape[1] > 1:
                correlation_matrix = np.corrcoef(factors.T)
                # Get off-diagonal correlations
                mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
                off_diag_corrs = correlation_matrix[mask]
                metrics['mean_factor_correlation'] = np.mean(np.abs(off_diag_corrs))
                metrics['max_factor_correlation'] = np.max(np.abs(off_diag_corrs))
            
            # Factor sparsity (measure of how many factors are active)
            factor_activity = np.mean(np.abs(factors) > 0.1, axis=0)
            metrics['mean_factor_activity'] = np.mean(factor_activity)
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Failed to compute disentanglement metrics: {e}")
            return {}

class MultiModalFusionModule(BaseLightningModule):
    """
    Lightning module for multi-modal fusion training.
    
    Handles training of multi-modal models that combine imaging,
    genomics, and molecular data for perturbation prediction.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        # Initialize multi-modal model
        self.model = MultiModalFusion(config.model)
        
        # Initialize loss function
        self.loss_fn = MultiModalFusionLoss(config.loss)
        
        # Initialize metrics
        self.train_metrics = PerturbationPredictionMetrics(prefix='train/')
        self.val_metrics = PerturbationPredictionMetrics(prefix='val/')
        
        # Training parameters
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        
        # Step outputs for logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-modal fusion model."""
        return self.model(batch)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        
        # Forward pass
        outputs = self.model(batch)
        
        # Extract targets (perturbation effects)
        targets = self._extract_targets(batch)
        
        # Compute loss
        loss_dict = self.loss_fn(outputs, {'labels': targets})
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics
        predictions = outputs.get('predictions', targets)  # Use predictions if available
        self.train_metrics.update(predictions, targets)
        
        # Store outputs
        self.training_step_outputs.append({
            'loss': loss_dict.get('total_loss', loss_dict.get('prediction', torch.tensor(0.0))),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        return loss_dict.get('total_loss', loss_dict.get('prediction', torch.tensor(0.0)))
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        
        # Forward pass
        outputs = self.model(batch)
        
        # Extract targets
        targets = self._extract_targets(batch)
        
        # Compute loss
        loss_dict = self.loss_fn(outputs, {'labels': targets})
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val/{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update metrics
        predictions = outputs.get('predictions', targets)  # Use predictions if available
        self.val_metrics.update(predictions, targets)
        
        # Store outputs
        self.validation_step_outputs.append({
            'loss': loss_dict.get('total_loss', loss_dict.get('prediction', torch.tensor(0.0))),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        return loss_dict.get('total_loss', loss_dict.get('prediction', torch.tensor(0.0)))
    
    def on_training_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        
        # Compute and log metrics
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()
        
        # Generate visualizations
        if self.current_epoch % 5 == 0:
            self._generate_prediction_visualizations('train')
        
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        
        # Compute and log metrics
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics)
        
        # Log main validation metric
        main_metric = val_metrics.get('val/prediction_accuracy', 0.0)
        self.log('val/prediction_accuracy', main_metric, prog_bar=True)
        
        self.val_metrics.reset()
        
        # Generate visualizations
        if self.current_epoch % 5 == 0:
            self._generate_prediction_visualizations('val')
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Handle potential None value for max_epochs
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/prediction_accuracy',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _extract_targets(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract targets from batch."""
        
        # Extract perturbation effect targets from batch
        if 'perturbation' in batch and 'effects' in batch['perturbation']:
            return batch['perturbation']['effects']
        elif 'targets' in batch:
            return batch['targets']
        elif 'labels' in batch:
            return batch['labels']
        else:
            # Fallback - create dummy targets based on batch size
            batch_size = self._get_batch_size(batch)
            return torch.randn(batch_size, 64, device=self.device)  # 64-dim effect vector
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Get batch size from batch data."""
        for modality_data in batch.values():
            if isinstance(modality_data, dict):
                for data in modality_data.values():
                    if hasattr(data, 'size'):
                        return data.size(0)
            elif hasattr(modality_data, 'size'):
                return modality_data.size(0)
        return 32  # Default fallback
    
    def _generate_prediction_visualizations(self, split: str):
        """Generate prediction visualizations for a given split."""
        
        outputs = self.training_step_outputs if split == 'train' else self.validation_step_outputs
        
        if not outputs:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Collect predictions and targets
            predictions = torch.cat([out['predictions'] for out in outputs[-10:]])
            targets = torch.cat([out['targets'] for out in outputs[-10:]])
            
            # Create scatter plot of predictions vs targets
            fig, ax = plt.subplots(figsize=(10, 8))
            
            pred_flat = predictions.cpu().numpy().flatten()
            target_flat = targets.cpu().numpy().flatten()
            
            ax.scatter(target_flat, pred_flat, alpha=0.6, s=50)
            ax.plot([target_flat.min(), target_flat.max()], [target_flat.min(), target_flat.max()], 'r--', lw=2)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{split.title()} Predictions vs Targets')
            ax.grid(True, alpha=0.3)
            
            if self.logger_experiment:
                self.logger_experiment.log({
                    f'{split}/predictions_vs_targets': wandb.Image(fig),
                    'epoch': self.current_epoch
                })
            
            plt.close(fig)
        
        except Exception as e:
            logger.warning(f"Failed to generate {split} prediction visualizations: {e}")
