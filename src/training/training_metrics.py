"""
Training metrics for perturbation biology models.

Provides comprehensive metrics for evaluating causal discovery,
perturbation prediction, and multi-modal learning performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Lazy imports to avoid scipy issues
try:
    from torchmetrics import Metric, MetricCollection
    from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
    from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
    TORCHMETRICS_AVAILABLE = True
except Exception:
    TORCHMETRICS_AVAILABLE = False
    # Create dummy classes
    class Metric:
        def __init__(self, **kwargs):
            pass
        def compute(self):
            return torch.tensor(0.0)
        def update(self, *args, **kwargs):
            pass
        def reset(self):
            pass
        def to(self, device):
            return self
    
    class MetricCollection:
        def __init__(self, metrics):
            self.metrics = metrics
        def compute(self):
            return {name: torch.tensor(0.0) for name in self.metrics}
        def update(self, *args, **kwargs):
            pass
        def reset(self):
            pass
        def to(self, device):
            return self
    
    # Dummy metric classes
    Accuracy = Precision = Recall = F1Score = AUROC = Metric
    MeanSquaredError = MeanAbsoluteError = R2Score = Metric

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    def adjusted_rand_score(a, b):
        return 0.0
    def normalized_mutual_info_score(a, b):
        return 0.0
import logging
from typing import cast

logger = logging.getLogger(__name__)


class PerturbationPredictionMetrics(Metric):
    """
    Comprehensive metrics for perturbation effect prediction.

    Evaluates accuracy of predicting biological perturbation effects
    across different scales and modalities.
    """

    def __init__(self, prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

        # State variables for accumulating results
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.uncertainties: List[torch.Tensor] = []
        self.num_samples = torch.tensor(0, dtype=torch.long, device=torch.device('cpu'))

        # Individual metrics
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ):
        """Update metric state with new predictions and targets."""

        if predictions.shape != targets.shape:
            logger.warning(
                f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}"
            )
            return

        self.predictions.append(predictions.detach())
        self.targets.append(targets.detach())

        if uncertainties is not None:
            self.uncertainties.append(uncertainties.detach())

        self.num_samples += predictions.shape[0]

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all perturbation prediction metrics."""

        if len(self.predictions) == 0:
            return {}

        # Concatenate all predictions and targets
        all_predictions = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        metrics = {}

        # Basic regression metrics
        metrics[f"{self.prefix}mse"] = self.mse(all_predictions, all_targets)
        metrics[f"{self.prefix}mae"] = self.mae(all_predictions, all_targets)
        metrics[f"{self.prefix}rmse"] = torch.sqrt(metrics[f"{self.prefix}mse"])

        # R-squared
        try:
            metrics[f"{self.prefix}r2"] = self.r2(all_predictions, all_targets)
        except:
            metrics[f"{self.prefix}r2"] = torch.tensor(0.0)

        # Pearson correlation
        metrics[f"{self.prefix}pearson_corr"] = self._compute_pearson_correlation(
            all_predictions, all_targets
        )

        # Effect magnitude accuracy (for perturbation biology)
        metrics[
            f"{self.prefix}effect_magnitude_accuracy"
        ] = self._compute_effect_magnitude_accuracy(all_predictions, all_targets)

        # Direction accuracy (whether effect is positive or negative)
        metrics[f"{self.prefix}direction_accuracy"] = self._compute_direction_accuracy(
            all_predictions, all_targets
        )

        # Uncertainty calibration (if uncertainties provided)
        if len(self.uncertainties) > 0:
            all_uncertainties = torch.cat(self.uncertainties, dim=0)
            metrics[
                f"{self.prefix}uncertainty_calibration"
            ] = self._compute_uncertainty_calibration(
                all_predictions, all_targets, all_uncertainties
            )

        # Top-k accuracy for ranking perturbations
        for k in [1, 5, 10]:
            metrics[f"{self.prefix}top_{k}_accuracy"] = self._compute_top_k_accuracy(
                all_predictions, all_targets, k
            )

        return metrics

    def _compute_pearson_correlation(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Pearson correlation coefficient."""

        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # Center the data
        pred_centered = predictions_flat - predictions_flat.mean()
        target_centered = targets_flat - targets_flat.mean()

        # Compute correlation
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum())

        if denominator == 0:
            return torch.tensor(0.0)

        correlation = numerator / denominator
        return correlation

    def _compute_effect_magnitude_accuracy(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute accuracy of effect magnitude prediction."""

        # Compute relative error in magnitude
        pred_magnitude = torch.abs(predictions)
        target_magnitude = torch.abs(targets)

        # Avoid division by zero
        relative_error = torch.abs(pred_magnitude - target_magnitude) / (target_magnitude + 1e-8)

        # Consider accurate if relative error < 20%
        accuracy = (relative_error < 0.2).float().mean()

        return accuracy

    def _compute_direction_accuracy(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute accuracy of effect direction (sign) prediction."""

        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)

        # Accuracy of sign prediction
        direction_accuracy = (pred_signs == target_signs).float().mean()

        return direction_accuracy

    def _compute_uncertainty_calibration(
        self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty calibration score."""

        # Compute prediction errors
        errors = torch.abs(predictions - targets)

        # Uncertainty should correlate with errors
        uncertainty_correlation = self._compute_pearson_correlation(
            uncertainties.flatten(), errors.flatten()
        )

        return uncertainty_correlation

    def _compute_top_k_accuracy(
        self, predictions: torch.Tensor, targets: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Compute top-k accuracy for perturbation ranking."""

        if predictions.numel() < k:
            return torch.tensor(0.0)

        # Get top-k predictions and targets
        _, pred_topk_indices = torch.topk(predictions.flatten(), k)
        _, target_topk_indices = torch.topk(targets.flatten(), k)

        # Compute intersection
        intersection = len(
            set(pred_topk_indices.cpu().numpy()) & set(target_topk_indices.cpu().numpy())
        )

        return torch.tensor(intersection / k)


class CausalDiscoveryMetrics(Metric):
    """
    Metrics for evaluating causal discovery performance.

    Evaluates the accuracy of discovered causal graphs and mechanisms.
    """

    def __init__(self, prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

        # State variables
        self.discovered_graphs: List[torch.Tensor] = []
        self.true_graphs: List[torch.Tensor] = []
        self.causal_factors: List[torch.Tensor] = []

    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Update metric state with causal discovery results."""

        if "causal_graph" in outputs:
            self.discovered_graphs.append(outputs["causal_graph"].detach())

        if "true_causal_graph" in targets:
            self.true_graphs.append(targets["true_causal_graph"].detach())

        if "causal_factors" in outputs:
            self.causal_factors.append(outputs["causal_factors"].detach())

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute causal discovery metrics."""

        if len(self.discovered_graphs) == 0:
            return {}

        metrics = {}

        # Graph structure metrics
        if len(self.true_graphs) > 0:
            structure_metrics = self._compute_graph_structure_metrics()
            metrics.update(structure_metrics)

        # Causal factor metrics
        if len(self.causal_factors) > 0:
            factor_metrics = self._compute_causal_factor_metrics()
            metrics.update(factor_metrics)

        return metrics

    def _compute_graph_structure_metrics(self) -> Dict[str, torch.Tensor]:
        """Compute metrics for causal graph structure accuracy."""

        metrics = {}

        # Average metrics across all graphs
        precisions, recalls, f1_scores = [], [], []
        shd_scores = []  # Structural Hamming Distance

        for discovered, true_graph in zip(self.discovered_graphs, self.true_graphs):
            # Convert to binary adjacency matrices
            discovered_binary = (discovered > 0.5).float()
            true_binary = true_graph.float()

            # Compute precision, recall, F1 for edge detection
            tp = (discovered_binary * true_binary).sum()
            fp = (discovered_binary * (1 - true_binary)).sum()
            fn = ((1 - discovered_binary) * true_binary).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            # Structural Hamming Distance
            shd = torch.sum(torch.abs(discovered_binary - true_binary))
            shd_scores.append(shd)

        metrics[f"{self.prefix}edge_precision"] = torch.stack(precisions).mean()
        metrics[f"{self.prefix}edge_recall"] = torch.stack(recalls).mean()
        metrics[f"{self.prefix}edge_f1"] = torch.stack(f1_scores).mean()
        metrics[f"{self.prefix}structural_hamming_distance"] = torch.stack(shd_scores).mean()

        # Graph-level metrics
        metrics[f"{self.prefix}graph_density"] = self._compute_average_graph_density()
        metrics[f"{self.prefix}causal_score"] = metrics[f"{self.prefix}edge_f1"]  # Main metric

        return metrics

    def _compute_causal_factor_metrics(self) -> Dict[str, torch.Tensor]:
        """Compute metrics for causal factor quality."""

        metrics = {}

        # Concatenate all causal factors
        all_factors = torch.cat(self.causal_factors, dim=0)

        # Factor diversity (average pairwise distance)
        factor_diversity = self._compute_factor_diversity(all_factors)
        metrics[f"{self.prefix}factor_diversity"] = factor_diversity

        # Factor stability (consistency across batches)
        if len(self.causal_factors) > 1:
            factor_stability = self._compute_factor_stability()
            metrics[f"{self.prefix}factor_stability"] = factor_stability

        return metrics

    def _compute_average_graph_density(self) -> torch.Tensor:
        """Compute average density of discovered graphs."""

        densities = []
        for graph in self.discovered_graphs:
            binary_graph = (graph > 0.5).float()
            num_edges = binary_graph.sum()
            num_possible_edges = graph.shape[0] * (graph.shape[0] - 1)  # Directed graph
            density = num_edges / num_possible_edges if num_possible_edges > 0 else 0
            densities.append(density)

        return torch.stack(densities).mean()

    def _compute_factor_diversity(self, factors: torch.Tensor) -> torch.Tensor:
        """Compute diversity of causal factors."""

        # Compute pairwise cosine similarities
        factors_norm = torch.nn.functional.normalize(factors, dim=1)
        similarity_matrix = torch.mm(factors_norm, factors_norm.t())

        # Remove diagonal (self-similarity)
        mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
        similarities = similarity_matrix[mask]

        # Diversity is inverse of average similarity
        diversity = 1.0 - similarities.mean()

        return diversity

    def _compute_factor_stability(self) -> torch.Tensor:
        """Compute stability of causal factors across batches."""

        if len(self.causal_factors) < 2:
            return torch.tensor(1.0)

        # Compute correlation between factors from different batches
        correlations = []

        for i in range(len(self.causal_factors) - 1):
            factors1 = self.causal_factors[i]
            factors2 = self.causal_factors[i + 1]

            # Compute correlation for each factor dimension
            for dim in range(factors1.shape[1]):
                corr = torch.corrcoef(torch.stack([factors1[:, dim], factors2[:, dim]]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(torch.abs(corr))

        if correlations:
            stability = torch.stack(correlations).mean()
        else:
            stability = torch.tensor(0.0)

        return stability


class ClassificationMetrics(Metric):
    """Standard classification metrics for cellular analysis."""

    def __init__(self, num_classes: int = 5, prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.num_classes = num_classes

        # State variables
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.probabilities: List[torch.Tensor] = []

        # Individual metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        if num_classes == 2:
            self.auroc = AUROC(task="binary")
        else:
            self.auroc = AUROC(task="multiclass", num_classes=num_classes)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
    ):
        """Update metric state with new predictions and targets."""

        self.predictions.append(predictions.detach())
        self.targets.append(targets.detach())

        if probabilities is not None:
            self.probabilities.append(probabilities.detach())

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all classification metrics."""

        if len(self.predictions) == 0:
            return {}

        # Concatenate all predictions and targets
        all_predictions = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        metrics = {}

        # Basic classification metrics
        metrics[f"{self.prefix}accuracy"] = self.accuracy(all_predictions, all_targets)
        metrics[f"{self.prefix}precision"] = self.precision(all_predictions, all_targets)
        metrics[f"{self.prefix}recall"] = self.recall(all_predictions, all_targets)
        metrics[f"{self.prefix}f1"] = self.f1(all_predictions, all_targets)

        # AUROC (if probabilities available)
        if len(self.probabilities) > 0:
            all_probabilities = torch.cat(self.probabilities, dim=0)
            try:
                metrics[f"{self.prefix}auroc"] = self.auroc(all_probabilities, all_targets)
            except:
                logger.warning("Failed to compute AUROC")
                metrics[f"{self.prefix}auroc"] = torch.tensor(0.0)

        # Confusion matrix statistics
        confusion_stats = self._compute_confusion_matrix_stats(all_predictions, all_targets)
        metrics.update(confusion_stats)

        return metrics

    def _compute_confusion_matrix_stats(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute statistics from confusion matrix."""

        metrics = {}

        # Compute confusion matrix
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

        for pred, target in zip(predictions, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                confusion_matrix[target.long(), pred.long()] += 1

        # Per-class metrics
        for class_idx in range(self.num_classes):
            tp = confusion_matrix[class_idx, class_idx]
            fp = confusion_matrix[:, class_idx].sum() - tp
            fn = confusion_matrix[class_idx, :].sum() - tp

            # Per-class precision and recall
            class_precision = tp.float() / (tp + fp + 1e-8)
            class_recall = tp.float() / (tp + fn + 1e-8)

            metrics[f"{self.prefix}precision_class_{class_idx}"] = class_precision
            metrics[f"{self.prefix}recall_class_{class_idx}"] = class_recall

        return metrics


class UncertaintyMetrics(Metric):
    """Metrics for evaluating uncertainty quantification."""

    def __init__(self, prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

        # State variables
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.uncertainties: List[torch.Tensor] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor):
        """Update metric state with predictions, targets, and uncertainties."""

        self.predictions.append(predictions.detach())
        self.targets.append(targets.detach())
        self.uncertainties.append(uncertainties.detach())

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute uncertainty quantification metrics."""

        if len(self.predictions) == 0:
            return {}

        # Concatenate all data
        all_predictions = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        all_uncertainties = torch.cat(self.uncertainties, dim=0)

        metrics = {}

        # Calibration metrics
        metrics[f"{self.prefix}calibration_error"] = self._compute_calibration_error(
            all_predictions, all_targets, all_uncertainties
        )

        # Uncertainty-error correlation
        metrics[f"{self.prefix}uncertainty_correlation"] = self._compute_uncertainty_correlation(
            all_predictions, all_targets, all_uncertainties
        )

        # Sharpness (average uncertainty)
        metrics[f"{self.prefix}sharpness"] = all_uncertainties.mean()

        # Coverage probability
        metrics[f"{self.prefix}coverage_probability"] = self._compute_coverage_probability(
            all_predictions, all_targets, all_uncertainties
        )

        return metrics

    def _compute_calibration_error(
        self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Compute calibration error for uncertainty estimates."""

        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainties.flatten())
        sorted_predictions = predictions.flatten()[sorted_indices]
        sorted_targets = targets.flatten()[sorted_indices]
        sorted_uncertainties = uncertainties.flatten()[sorted_indices]

        # Divide into bins
        num_bins = 10
        bin_size = len(sorted_predictions) // num_bins

        calibration_errors = []

        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_predictions)

            if start_idx >= end_idx:
                continue

            bin_predictions = sorted_predictions[start_idx:end_idx]
            bin_targets = sorted_targets[start_idx:end_idx]
            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]

            # Expected error based on uncertainty
            expected_error = bin_uncertainties.mean()

            # Actual error
            actual_error = torch.abs(bin_predictions - bin_targets).mean()

            # Calibration error for this bin
            calibration_error = torch.abs(expected_error - actual_error)
            calibration_errors.append(calibration_error)

        return torch.stack(calibration_errors).mean() if calibration_errors else torch.tensor(0.0)

    def _compute_uncertainty_correlation(
        self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Compute correlation between uncertainty and prediction error."""

        errors = torch.abs(predictions - targets).flatten()
        uncertainties_flat = uncertainties.flatten()

        # Compute Pearson correlation
        errors_centered = errors - errors.mean()
        uncertainties_centered = uncertainties_flat - uncertainties_flat.mean()

        numerator = (errors_centered * uncertainties_centered).sum()
        denominator = torch.sqrt((errors_centered**2).sum() * (uncertainties_centered**2).sum())

        if denominator == 0:
            return torch.tensor(0.0)

        correlation = numerator / denominator
        return correlation

    def _compute_coverage_probability(
        self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Compute coverage probability for uncertainty intervals."""

        # Assume uncertainties represent standard deviations
        # Compute 95% confidence intervals
        confidence_level = 1.96  # 95% confidence for normal distribution

        lower_bounds = predictions - confidence_level * uncertainties
        upper_bounds = predictions + confidence_level * uncertainties

        # Check if true values fall within confidence intervals
        within_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
        coverage_probability = within_interval.float().mean()

        return coverage_probability


class BiologicalInterpretabilityMetrics(Metric):
    """Metrics for evaluating biological interpretability of models."""

    def __init__(self, prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

        # State variables for biological annotations
        self.attention_maps: List[torch.Tensor] = []
        self.concept_activations: List[torch.Tensor] = []
        self.pathway_scores: List[torch.Tensor] = []
        self.biological_labels: List[torch.Tensor] = []

    def update(
        self,
        attention_maps: Optional[torch.Tensor] = None,
        concept_activations: Optional[torch.Tensor] = None,
        pathway_scores: Optional[torch.Tensor] = None,
        biological_labels: Optional[torch.Tensor] = None,
    ):
        """Update metric state with interpretability data."""

        if attention_maps is not None:
            self.attention_maps.append(attention_maps.detach())

        if concept_activations is not None:
            self.concept_activations.append(concept_activations.detach())

        if pathway_scores is not None:
            self.pathway_scores.append(pathway_scores.detach())

        if biological_labels is not None:
            self.biological_labels.append(biological_labels.detach())

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute biological interpretability metrics."""

        metrics = {}

        # Attention map interpretability
        if len(self.attention_maps) > 0:
            attention_metrics = self._compute_attention_interpretability()
            metrics.update(attention_metrics)

        # Concept activation interpretability
        if len(self.concept_activations) > 0:
            concept_metrics = self._compute_concept_interpretability()
            metrics.update(concept_metrics)

        # Pathway enrichment interpretability
        if len(self.pathway_scores) > 0:
            pathway_metrics = self._compute_pathway_interpretability()
            metrics.update(pathway_metrics)

        return metrics

    def _compute_attention_interpretability(self) -> Dict[str, torch.Tensor]:
        """Compute interpretability metrics for attention maps."""

        metrics = {}

        # Concatenate all attention maps
        all_attention = torch.cat(self.attention_maps, dim=0)

        # Attention sparsity (how focused is the attention)
        attention_entropy = self._compute_attention_entropy(all_attention)
        metrics[f"{self.prefix}attention_entropy"] = attention_entropy

        # Attention consistency (similarity across samples)
        attention_consistency = self._compute_attention_consistency(all_attention)
        metrics[f"{self.prefix}attention_consistency"] = attention_consistency

        return metrics

    def _compute_concept_interpretability(self) -> Dict[str, torch.Tensor]:
        """Compute interpretability metrics for concept activations."""

        metrics = {}

        # Concatenate all concept activations
        all_concepts = torch.cat(self.concept_activations, dim=0)

        # Concept selectivity (how selective are the concepts)
        concept_selectivity = self._compute_concept_selectivity(all_concepts)
        metrics[f"{self.prefix}concept_selectivity"] = concept_selectivity

        # Concept diversity (how diverse are the activated concepts)
        concept_diversity = self._compute_concept_diversity(all_concepts)
        metrics[f"{self.prefix}concept_diversity"] = concept_diversity

        return metrics

    def _compute_pathway_interpretability(self) -> Dict[str, torch.Tensor]:
        """Compute interpretability metrics for pathway scores."""

        metrics = {}

        # Concatenate all pathway scores
        all_pathways = torch.cat(self.pathway_scores, dim=0)

        # Pathway enrichment strength
        pathway_enrichment = all_pathways.mean()
        metrics[f"{self.prefix}pathway_enrichment"] = pathway_enrichment

        # Number of significantly enriched pathways
        significant_pathways = (all_pathways > 0.05).float().mean()  # Assuming scores are p-values
        metrics[f"{self.prefix}significant_pathways"] = significant_pathways

        return metrics

    def _compute_attention_entropy(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention maps."""

        # Flatten attention maps
        attention_flat = attention_maps.view(attention_maps.size(0), -1)

        # Normalize to probabilities
        attention_probs = torch.softmax(attention_flat, dim=1)

        # Compute entropy
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=1)

        return entropy.mean()

    def _compute_attention_consistency(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """Compute consistency of attention maps across samples."""

        # Flatten attention maps
        attention_flat = attention_maps.view(attention_maps.size(0), -1)

        # Normalize attention maps
        attention_norm = torch.nn.functional.normalize(attention_flat, dim=1)

        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(attention_norm, attention_norm.t())

        # Remove diagonal (self-similarity) and compute average
        mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        consistency = similarity_matrix[mask].mean()

        return consistency

    def _compute_concept_selectivity(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """Compute selectivity of concept activations."""

        # Compute sparsity of activations (higher sparsity = more selective)
        activation_sparsity = torch.zeros(concept_activations.size(1))

        for concept_idx in range(concept_activations.size(1)):
            concept_values = concept_activations[:, concept_idx]
            # Compute Gini coefficient as sparsity measure
            sorted_values, _ = torch.sort(concept_values)
            n = len(sorted_values)
            cumsum = torch.cumsum(sorted_values, dim=0)
            gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
            activation_sparsity[concept_idx] = gini

        return activation_sparsity.mean()

    def _compute_concept_diversity(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """Compute diversity of concept activations."""

        # Compute correlation matrix between concepts
        concepts_centered = concept_activations - concept_activations.mean(dim=0)
        correlation_matrix = torch.corrcoef(concepts_centered.t())

        # Remove diagonal and compute average absolute correlation
        mask = ~torch.eye(correlation_matrix.size(0), dtype=torch.bool, device=correlation_matrix.device)
        avg_correlation = torch.abs(correlation_matrix[mask]).mean()

        # Diversity is inverse of correlation
        diversity = 1.0 - avg_correlation

        return diversity


class OpenPerturbationMetricCollection(object):
    """Collection of metrics for comprehensive model evaluation."""

    def __init__(self, config: Dict):
        self.config = config
        self.metrics: Dict[str, Metric] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize all metrics based on configuration."""

        # Perturbation prediction metrics
        if self.config.get("perturbation_prediction", True):
            self.metrics["perturbation"] = PerturbationPredictionMetrics(prefix="perturbation/")

        # Causal discovery metrics
        if self.config.get("causal_discovery", True):
            self.metrics["causal"] = CausalDiscoveryMetrics(prefix="causal/")

        # Classification metrics (for cell cycle, health, etc.)
        if self.config.get("classification", True):
            num_classes = self.config.get("num_classes", 5)
            self.metrics["classification"] = ClassificationMetrics(
                num_classes=num_classes, prefix="classification/"
            )

        # Uncertainty quantification metrics
        if self.config.get("uncertainty", True):
            self.metrics["uncertainty"] = UncertaintyMetrics(prefix="uncertainty/")

        # Biological interpretability metrics
        if self.config.get("interpretability", True):
            self.metrics["interpretability"] = BiologicalInterpretabilityMetrics(
                prefix="interpretability/"
            )

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ):
        """Update all metrics with new predictions and targets."""

        # Update perturbation prediction metrics
        if "perturbation" in self.metrics and "perturbation_predictions" in outputs:
            perturbation_targets = targets.get("perturbation_effects", targets.get("effects"))
            uncertainties = outputs.get("perturbation_uncertainties")

            self.metrics["perturbation"].update(
                outputs["perturbation_predictions"], perturbation_targets, uncertainties
            )

        # Update causal discovery metrics
        if "causal" in self.metrics:
            self.metrics["causal"].update(outputs, targets)

        # Update classification metrics
        if "classification" in self.metrics and "classification_logits" in outputs:
            classification_targets = targets.get("classification_labels", targets.get("labels"))
            classification_probs = torch.softmax(outputs["classification_logits"], dim=1)
            classification_preds = torch.argmax(classification_probs, dim=1)

            self.metrics["classification"].update(
                classification_preds, classification_targets, classification_probs
            )

        # Update uncertainty metrics
        if "uncertainty" in self.metrics and "uncertainty" in outputs:
            main_predictions = outputs.get("perturbation_predictions", outputs.get("predictions"))
            main_targets = targets.get("perturbation_effects", targets.get("effects"))

            if main_predictions is not None and main_targets is not None:
                self.metrics["uncertainty"].update(
                    main_predictions, main_targets, outputs["uncertainty"]
                )

        # Update interpretability metrics
        if "interpretability" in self.metrics:
            self.metrics["interpretability"].update(
                attention_maps=outputs.get("attention_maps"),
                concept_activations=outputs.get("concept_activations"),
                pathway_scores=outputs.get("pathway_scores"),
                biological_labels=targets.get("biological_labels"),
            )

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics and return results."""

        all_metrics: Dict[str, torch.Tensor] = {}

        for metric_name, metric in self.metrics.items():
            try:
                metric_results = metric.compute()
                all_metrics.update(metric_results)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name} metrics: {e}")

        return all_metrics

    def reset(self):
        """Reset all metrics."""

        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        """Move all metrics to device."""

        for metric in self.metrics.values():
            metric.to(device)

        return self


# Utility functions for metric computation


def compute_biological_pathway_enrichment(
    gene_scores: torch.Tensor, pathway_genes: Dict[str, List[int]], method: str = "gsea"
) -> Dict[str, float]:
    """
    Compute pathway enrichment scores for given gene scores.

    Args:
        gene_scores: Tensor of shape [num_genes] with scores for each gene
        pathway_genes: Dictionary mapping pathway names to lists of gene indices
        method: Enrichment method ('gsea', 'fisher', 'hypergeometric')

    Returns:
        Dictionary mapping pathway names to enrichment scores
    """

    enrichment_scores = {}
    gene_scores_np = gene_scores.cpu().numpy()

    for pathway_name, gene_indices in pathway_genes.items():
        if method == "gsea":
            # Gene Set Enrichment Analysis
            score = compute_gsea_score(gene_scores_np, gene_indices)
        elif method == "fisher":
            # Fisher's exact test
            score = compute_fisher_enrichment(gene_scores_np, gene_indices)
        elif method == "hypergeometric":
            # Hypergeometric test
            score = compute_hypergeometric_enrichment(gene_scores_np, gene_indices)
        else:
            raise ValueError(f"Unknown enrichment method: {method}")

        enrichment_scores[pathway_name] = score

    return enrichment_scores


def compute_gsea_score(gene_scores: np.ndarray, pathway_genes: List[int]) -> float:
    """Compute Gene Set Enrichment Analysis score."""

    # Rank genes by score
    ranked_indices = np.argsort(gene_scores)[::-1]  # Descending order

    # Compute enrichment score
    n_genes = len(gene_scores)
    n_pathway = len(pathway_genes)

    pathway_set = set(pathway_genes)
    running_sum = 0.0
    max_deviation = 0.0

    # Calculate sum of scores for pathway genes
    pathway_score_sum = sum(abs(gene_scores[i]) for i in pathway_genes if i < len(gene_scores))

    for gene_idx in ranked_indices:
        if gene_idx in pathway_set:
            # Hit: add to running sum
            if pathway_score_sum > 0:
                running_sum += abs(gene_scores[gene_idx]) / pathway_score_sum
        else:
            # Miss: subtract from running sum
            running_sum -= 1.0 / (n_genes - n_pathway)

        # Track maximum deviation
        if abs(running_sum) > abs(max_deviation):
            max_deviation = running_sum

    # Cast to builtin float to satisfy static type checkers
    return float(max_deviation)


def compute_fisher_enrichment(
    gene_scores: np.ndarray, pathway_genes: List[int], threshold: float = 0.5
) -> float:
    """Compute Fisher's exact test enrichment score."""

    try:
        from scipy.stats import fisher_exact

        # Binarize gene scores
        significant_genes = gene_scores > threshold

        # Create contingency table
        pathway_set = set(pathway_genes)

        # Count genes in each category
        pathway_significant = sum(
            1 for i in pathway_genes if i < len(gene_scores) and significant_genes[i]
        )
        pathway_not_significant = sum(
            1 for i in pathway_genes if i < len(gene_scores) and not significant_genes[i]
        )
        non_pathway_significant = np.sum(significant_genes) - pathway_significant
        non_pathway_not_significant = (
            len(gene_scores)
            - pathway_not_significant
            - non_pathway_significant
            - pathway_significant
        )

        # Contingency table
        table = [
            [pathway_significant, pathway_not_significant],
            [non_pathway_significant, non_pathway_not_significant],
        ]

        # Fisher's exact test
        odds_ratio, p_value = fisher_exact(table, alternative="greater")

        # Convert to builtin float; use fallback for non-scalar cases
        try:
            pval: float = float(p_value)  # type: ignore[arg-type]
        except Exception:
            pval = float(np.asarray(p_value).item())

        return -np.log10(max(pval, 1e-10))

    except ImportError:
        logger.warning("scipy not available for Fisher's exact test")
        return 0.0


def compute_hypergeometric_enrichment(
    gene_scores: np.ndarray, pathway_genes: List[int], threshold: float = 0.5
) -> float:
    """Compute hypergeometric test enrichment score."""

    try:
        from scipy.stats import hypergeom

        # Binarize gene scores
        significant_genes = gene_scores > threshold

        # Parameters for hypergeometric distribution
        M = len(gene_scores)  # Total number of genes
        n = len(pathway_genes)  # Number of pathway genes
        N = np.sum(significant_genes)  # Number of significant genes

        # Number of significant genes in pathway
        k = sum(1 for i in pathway_genes if i < len(gene_scores) and significant_genes[i])

        # Hypergeometric test (probability of observing k or more successes)
        p_value = hypergeom.sf(k - 1, M, n, N)

        # Convert to builtin float; use fallback for non-scalar cases
        try:
            pval: float = float(p_value)  # type: ignore[arg-type]
        except Exception:
            pval = float(np.asarray(p_value).item())

        return -np.log10(max(pval, 1e-10))

    except ImportError:
        logger.warning("scipy not available for hypergeometric test")
        return 0.0


def compute_attention_localization_score(
    attention_map: torch.Tensor, ground_truth_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute localization score for attention maps.

    Measures how well attention maps localize to regions of interest.
    """

    # Normalize attention map
    attention_norm = attention_map / (attention_map.sum() + 1e-8)

    # Compute overlap with ground truth
    overlap = (attention_norm * ground_truth_mask).sum()

    return overlap


def compute_model_complexity_metrics(model: torch.nn.Module) -> Dict[str, float]:
    """Compute complexity metrics for a PyTorch model."""

    metrics = {}

    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics["total_parameters"] = total_params
    metrics["trainable_parameters"] = trainable_params
    metrics["parameter_efficiency"] = trainable_params / total_params if total_params > 0 else 0

    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024**2)

    metrics["model_size_mb"] = model_size_mb

    # Compute sparsity if model has sparse layers
    sparse_params = 0
    total_weights = 0

    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None:
            weight_tensor = cast(torch.Tensor, module.weight.data)
            total_weights += int(weight_tensor.numel())
            sparsity_mask = torch.lt(torch.abs(weight_tensor), 1e-6)
            sparse_params += int(sparsity_mask.sum().item())

    if total_weights > 0:
        metrics["sparsity"] = sparse_params / total_weights

    return metrics


# Testing functions


def test_perturbation_prediction_metrics():
    """Test perturbation prediction metrics."""

    print("TEST: Testing Perturbation Prediction Metrics...")

    # Create dummy data
    batch_size = 32
    predictions = torch.randn(batch_size, 10)
    targets = torch.randn(batch_size, 10)
    uncertainties = torch.abs(torch.randn(batch_size, 10))

    # Initialize metrics
    metrics = PerturbationPredictionMetrics()

    # Update metrics
    metrics.update(predictions, targets, uncertainties)

    # Compute metrics
    results = metrics.compute()

    print(f"  SUCCESS: MSE: {results['mse']:.4f}")
    print(f"  SUCCESS: MAE: {results['mae']:.4f}")
    print(f"  SUCCESS: R²: {results['r2']:.4f}")
    print(f"  SUCCESS: Pearson Correlation: {results['pearson_corr']:.4f}")
    print(f"  SUCCESS: Effect Magnitude Accuracy: {results['effect_magnitude_accuracy']:.4f}")


def test_causal_discovery_metrics():
    """Test causal discovery metrics."""

    print("TEST: Testing Causal Discovery Metrics...")

    # Create dummy causal graphs
    batch_size = 4
    num_factors = 8

    discovered_graphs = torch.sigmoid(torch.randn(batch_size, num_factors, num_factors))
    true_graphs = torch.randint(0, 2, (batch_size, num_factors, num_factors)).float()
    causal_factors = torch.randn(batch_size, 16, num_factors)

    # Initialize metrics
    metrics = CausalDiscoveryMetrics()

    # Update metrics
    for i in range(batch_size):
        outputs = {
            "causal_graph": discovered_graphs[i : i + 1],
            "causal_factors": causal_factors[i : i + 1],
        }
        targets = {"true_causal_graph": true_graphs[i : i + 1]}
        metrics.update(outputs, targets)

    # Compute metrics
    results = metrics.compute()

    print(f"  SUCCESS: Edge Precision: {results['causal/edge_precision']:.4f}")
    print(f"  SUCCESS: Edge Recall: {results['causal/edge_recall']:.4f}")
    print(f"  SUCCESS: Edge F1: {results['causal/edge_f1']:.4f}")
    print(f"  SUCCESS: Causal Score: {results['causal/causal_score']:.4f}")


def test_classification_metrics():
    """Test classification metrics."""

    print("TEST: Testing Classification Metrics...")

    # Create dummy classification data
    batch_size = 64
    num_classes = 5

    predictions = torch.randint(0, num_classes, (batch_size,))
    targets = torch.randint(0, num_classes, (batch_size,))
    probabilities = torch.softmax(torch.randn(batch_size, num_classes), dim=1)

    # Initialize metrics
    metrics = ClassificationMetrics(num_classes=num_classes)

    # Update metrics
    metrics.update(predictions, targets, probabilities)

    # Compute metrics
    results = metrics.compute()

    print(f"  SUCCESS: Accuracy: {results['accuracy']:.4f}")
    print(f"  SUCCESS: Precision: {results['precision']:.4f}")
    print(f"  SUCCESS: Recall: {results['recall']:.4f}")
    print(f"  SUCCESS: F1 Score: {results['f1']:.4f}")


def test_uncertainty_metrics():
    """Test uncertainty quantification metrics."""

    print("TEST: Testing Uncertainty Metrics...")

    # Create dummy data with correlated uncertainties and errors
    batch_size = 100
    predictions = torch.randn(batch_size)
    targets = torch.randn(batch_size)

    # Make uncertainties correlate with prediction errors
    errors = torch.abs(predictions - targets)
    uncertainties = errors + 0.1 * torch.randn(batch_size)
    uncertainties = torch.abs(uncertainties)  # Ensure positive

    # Initialize metrics
    metrics = UncertaintyMetrics()

    # Update metrics
    metrics.update(predictions, targets, uncertainties)

    # Compute metrics
    results = metrics.compute()

    print(f"  SUCCESS: Calibration Error: {results['calibration_error']:.4f}")
    print(f"  SUCCESS: Uncertainty Correlation: {results['uncertainty_correlation']:.4f}")
    print(f"  SUCCESS: Coverage Probability: {results['coverage_probability']:.4f}")


def test_interpretability_metrics():
    """Test biological interpretability metrics."""

    print("TEST: Testing Interpretability Metrics...")

    # Create dummy interpretability data
    batch_size = 16
    attention_size = 196  # 14x14 patches
    num_concepts = 50
    num_pathways = 100

    attention_maps = torch.softmax(torch.randn(batch_size, attention_size), dim=1)
    concept_activations = torch.randn(batch_size, num_concepts)
    pathway_scores = torch.rand(batch_size, num_pathways)  # p-values

    # Initialize metrics
    metrics = BiologicalInterpretabilityMetrics()

    # Update metrics
    metrics.update(
        attention_maps=attention_maps,
        concept_activations=concept_activations,
        pathway_scores=pathway_scores,
    )

    # Compute metrics
    results = metrics.compute()

    print(f"  SUCCESS: Attention Entropy: {results['interpretability/attention_entropy']:.4f}")
    print(f"  SUCCESS: Concept Selectivity: {results['interpretability/concept_selectivity']:.4f}")
    print(f"  SUCCESS: Pathway Enrichment: {results['interpretability/pathway_enrichment']:.4f}")


def test_metric_collection():
    """Test the complete metric collection."""

    print("TEST: Testing Metric Collection...")

    # Configuration for metrics
    config = {
        "perturbation_prediction": True,
        "causal_discovery": True,
        "classification": True,
        "uncertainty": True,
        "interpretability": True,
        "num_classes": 5,
    }

    # Initialize metric collection
    metric_collection = OpenPerturbationMetricCollection(config)

    # Create dummy batch data
    batch_size = 32
    outputs = {
        "perturbation_predictions": torch.randn(batch_size, 10),
        "perturbation_uncertainties": torch.abs(torch.randn(batch_size, 10)),
        "causal_graph": torch.sigmoid(torch.randn(batch_size, 8, 8)),
        "causal_factors": torch.randn(batch_size, 16, 8),
        "classification_logits": torch.randn(batch_size, 5),
        "uncertainty": torch.abs(torch.randn(batch_size, 10)),
        "attention_maps": torch.softmax(torch.randn(batch_size, 196), dim=1),
        "concept_activations": torch.randn(batch_size, 50),
        "pathway_scores": torch.rand(batch_size, 100),
    }

    targets = {
        "perturbation_effects": torch.randn(batch_size, 10),
        "true_causal_graph": torch.randint(0, 2, (batch_size, 8, 8)).float(),
        "classification_labels": torch.randint(0, 5, (batch_size,)),
        "effects": torch.randn(batch_size, 10),
    }

    batch = {}

    # Update metrics
    metric_collection.update(outputs, targets, batch)

    # Compute all metrics
    all_results = metric_collection.compute()

    print(f"  SUCCESS: Computed {len(all_results)} metrics successfully")
    for metric_name, value in list(all_results.items())[:10]:  # Show first 10
        print(f"    STATS: {metric_name}: {value:.4f}")


def run_all_metric_tests():
    """Run all metric tests."""

    print("TEST:" * 25)
    print("TESTING OPENPERTURBATIONS METRICS")
    print("TEST:" * 25)

    try:
        test_perturbation_prediction_metrics()
        print()

        test_causal_discovery_metrics()
        print()

        test_classification_metrics()
        print()

        test_uncertainty_metrics()
        print()

        test_interpretability_metrics()
        print()

        test_metric_collection()
        print()

        print("COMPLETE: All metric tests passed successfully!")

    except Exception as e:
        print(f" Metric tests failed: {e}")
        raise


if __name__ == "__main__":
    run_all_metric_tests()
