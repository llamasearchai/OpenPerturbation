"""Custom loss functions for perturbation biology models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class CausalConsistencyLoss(nn.Module):
    """Loss for enforcing causal consistency in perturbation models."""

    def __init__(self, lambda_causal: float = 1.0):
        super().__init__()
        self.lambda_causal = lambda_causal

    def forward(
        self,
        causal_factors: torch.Tensor,
        interventions: torch.Tensor,
        predicted_effects: torch.Tensor,
        actual_effects: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal consistency loss.

        Args:
            causal_factors: Extracted causal factors [B, D_causal]
            interventions: Intervention indicators [B, D_intervention]
            predicted_effects: Predicted intervention effects [B, D_effect]
            actual_effects: Actual observed effects [B, D_effect]
        """
        # Base prediction loss
        prediction_loss = F.mse_loss(predicted_effects, actual_effects)

        # Causal consistency: similar interventions should have similar effects
        consistency_loss = self._compute_consistency_loss(
            causal_factors, interventions, predicted_effects
        )

        # Independence constraint: causal factors should be independent
        independence_loss = self._compute_independence_loss(causal_factors)

        total_loss = (
            prediction_loss + self.lambda_causal * consistency_loss + 0.1 * independence_loss
        )

        return total_loss

    def _compute_consistency_loss(
        self,
        causal_factors: torch.Tensor,
        interventions: torch.Tensor,
        predicted_effects: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency loss for similar interventions."""
        batch_size = causal_factors.size(0)

        # Compute pairwise similarities in intervention space
        intervention_sim = torch.mm(interventions, interventions.t())
        intervention_sim = intervention_sim / (
            torch.norm(interventions, dim=1, keepdim=True) + 1e-8
        )

        # Compute pairwise similarities in effect space
        effect_sim = torch.mm(predicted_effects, predicted_effects.t())
        effect_sim = effect_sim / (torch.norm(predicted_effects, dim=1, keepdim=True) + 1e-8)

        # Consistency loss: similar interventions should have similar effects
        consistency_loss = F.mse_loss(intervention_sim, effect_sim)

        return consistency_loss

    def _compute_independence_loss(self, causal_factors: torch.Tensor) -> torch.Tensor:
        """Compute independence loss for causal factors."""
        # Center the factors
        centered_factors = causal_factors - causal_factors.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        cov_matrix = torch.mm(centered_factors.t(), centered_factors) / (causal_factors.size(0) - 1)

        # Independence loss: minimize off-diagonal elements
        identity = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
        independence_loss = torch.norm(cov_matrix - torch.diag(torch.diag(cov_matrix))) ** 2

        return independence_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning."""

    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, mode: str = "infonce"
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Feature embeddings [B, D]
            labels: Class labels [B]
            mode: 'infonce' or 'triplet'
        """
        if mode == "infonce":
            return self._compute_infonce_loss(embeddings, labels)
        elif mode == "triplet":
            return self._compute_triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown contrastive mode: {mode}")

    def _compute_infonce_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss."""
        batch_size = embeddings.size(0)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create positive mask
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.t()).float()

        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(batch_size, device=positive_mask.device)

        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)

        log_prob = similarity_matrix - torch.log(sum_exp_sim)

        # Average over positive pairs
        positive_log_prob = (positive_mask * log_prob).sum(dim=1)
        num_positives = positive_mask.sum(dim=1)

        # Avoid division by zero
        num_positives = torch.clamp(num_positives, min=1.0)
        loss = -(positive_log_prob / num_positives).mean()

        return loss

    def _compute_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss."""
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        negative_mask = (labels_expanded != labels_expanded.t()).float()

        # Remove diagonal
        positive_mask = positive_mask - torch.eye(batch_size, device=positive_mask.device)

        # Find hardest positive and negative examples
        positive_distances = distances * positive_mask + 1e6 * (1 - positive_mask)
        negative_distances = distances * negative_mask + 1e6 * negative_mask

        hardest_positive = positive_distances.min(dim=1)[0]
        hardest_negative = negative_distances.min(dim=1)[0]

        # Triplet loss
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)

        return triplet_loss.mean()


class UncertaintyLoss(nn.Module):
    """Loss function for uncertainty quantification."""

    def __init__(self, loss_type: str = "gaussian"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self, mean_pred: torch.Tensor, var_pred: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.

        Args:
            mean_pred: Predicted means [B, D]
            var_pred: Predicted variances [B, D]
            targets: Target values [B, D]
        """
        if self.loss_type == "gaussian":
            return self._gaussian_nll_loss(mean_pred, var_pred, targets)
        elif self.loss_type == "laplace":
            return self._laplace_nll_loss(mean_pred, var_pred, targets)
        else:
            raise ValueError(f"Unknown uncertainty loss type: {self.loss_type}")

    def _gaussian_nll_loss(
        self, mean_pred: torch.Tensor, var_pred: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood loss."""
        # Ensure positive variance
        var_pred = torch.clamp(var_pred, min=1e-6)

        # Compute NLL
        mse_term = (targets - mean_pred) ** 2 / var_pred
        log_var_term = torch.log(var_pred)
        constant_term = 0.5 * np.log(2 * np.pi)

        nll = 0.5 * (mse_term + log_var_term) + constant_term

        return nll.mean()

    def _laplace_nll_loss(
        self, mean_pred: torch.Tensor, scale_pred: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Laplace negative log-likelihood loss."""
        # Ensure positive scale
        scale_pred = torch.clamp(scale_pred, min=1e-6)

        # Compute NLL
        abs_error = torch.abs(targets - mean_pred)
        nll = abs_error / scale_pred + torch.log(2 * scale_pred)

        return nll.mean()


class StructuralLoss(nn.Module):
    """Loss functions for structural constraints in causal models."""

    def __init__(self, lambda_sparse: float = 0.1, lambda_dag: float = 1.0):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_dag = lambda_dag

    def forward(self, adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute structural losses for causal graphs.

        Args:
            adjacency_matrix: Learned adjacency matrix [D, D]
        """
        sparsity_loss = self._sparsity_loss(adjacency_matrix)
        dag_loss = self._dag_constraint_loss(adjacency_matrix)

        total_loss = self.lambda_sparse * sparsity_loss + self.lambda_dag * dag_loss

        return {"structural_loss": total_loss, "sparsity_loss": sparsity_loss, "dag_loss": dag_loss}

    def _sparsity_loss(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """L1 sparsity loss to encourage sparse graphs."""
        return torch.norm(adjacency_matrix, p=1)

    def _dag_constraint_loss(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """DAG constraint using matrix exponential trace."""
        d = adjacency_matrix.size(0)

        # Use polynomial approximation for matrix exponential
        A = adjacency_matrix
        powers = [torch.eye(d, device=A.device)]
        trace_sum = torch.tensor(float(d), device=A.device)  # tr(I) as tensor

        # Compute first few terms of exp(A)
        for i in range(1, 10):
            powers.append(torch.matmul(powers[-1], A) / i)
            trace_sum += torch.trace(powers[-1])

        # DAG constraint: tr(exp(A)) = d
        target_trace = torch.tensor(float(d), device=A.device)
        dag_constraint = torch.abs(trace_sum - target_trace)

        return dag_constraint


class BiologicalConsistencyLoss(nn.Module):
    """Loss function incorporating biological prior knowledge."""

    def __init__(self, pathway_graph: Optional[torch.Tensor] = None, lambda_pathway: float = 0.5):
        super().__init__()
        self.pathway_graph = pathway_graph
        self.lambda_pathway = lambda_pathway

        if pathway_graph is not None:
            self.register_buffer("pathway_adjacency", pathway_graph)

    def forward(
        self,
        predicted_graph: torch.Tensor,
        predicted_effects: torch.Tensor,
        pathway_effects: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute biological consistency loss.

        Args:
            predicted_graph: Predicted causal graph [D, D]
            predicted_effects: Predicted perturbation effects [B, D]
            pathway_effects: Known pathway effects [B, D]
        """
        total_loss = torch.tensor(0.0, device=predicted_graph.device)

        # Pathway structure consistency
        if self.pathway_graph is not None:
            structure_loss = self._pathway_structure_loss(predicted_graph)
            total_loss = total_loss + self.lambda_pathway * structure_loss

        # Pathway effect consistency
        if pathway_effects is not None:
            effect_loss = self._pathway_effect_loss(predicted_effects, pathway_effects)
            total_loss = total_loss + effect_loss

        return total_loss

    def _pathway_structure_loss(self, predicted_graph: torch.Tensor) -> torch.Tensor:
        """Encourage consistency with known pathway structure."""
        # MSE between predicted and known pathway connections
        pathway_adj = getattr(self, 'pathway_adjacency', None)
        if pathway_adj is not None and isinstance(pathway_adj, torch.Tensor):
            return F.mse_loss(predicted_graph, pathway_adj)
        else:
            # Fallback if pathway adjacency is not available
            return torch.tensor(0.0, device=predicted_graph.device)

    def _pathway_effect_loss(
        self, predicted_effects: torch.Tensor, pathway_effects: torch.Tensor
    ) -> torch.Tensor:
        """Encourage consistency with known pathway effects."""
        return F.mse_loss(predicted_effects, pathway_effects)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with automatic weighting."""

    def __init__(self, task_names: list, initial_weights: Optional[Dict] = None):
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)

        # Initialize learnable weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in task_names}

        # Log-space weights for stability
        self.log_weights = nn.Parameter(
            torch.tensor([np.log(initial_weights[name]) for name in task_names])
        )

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Args:
            task_losses: Dictionary of individual task losses
        """
        total_loss = 0.0
        weighted_losses = {}

        # Convert log weights to actual weights
        weights = torch.exp(self.log_weights)

        for i, task_name in enumerate(self.task_names):
            if task_name in task_losses:
                task_loss = task_losses[task_name]
                weight = weights[i]

                # Weighted loss with uncertainty-based weighting
                weighted_loss = task_loss / (2 * weight**2) + torch.log(weight)

                total_loss += weighted_loss
                weighted_losses[f"weighted_{task_name}"] = weighted_loss
                weighted_losses[f"weight_{task_name}"] = weight

        weighted_losses["total_loss"] = total_loss

        return weighted_losses
