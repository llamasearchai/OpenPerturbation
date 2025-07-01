"""
Concept Activation Vector (CAV) analysis for biological interpretability.

This module implements Testing with Concept Activation Vectors (TCAV) for
understanding what biological concepts learned models are using for predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Lazy imports for plotting - only import when needed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
from collections import defaultdict
import json
import warnings

# Set availability flags for lazy imports
SKLEARN_AVAILABLE = False
SCIPY_AVAILABLE = False

# Check sklearn availability
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except Exception:
    pass

# Check scipy availability  
try:
    import scipy
    SCIPY_AVAILABLE = True
except Exception:
    pass

logger = logging.getLogger(__name__)


@dataclass
class BiologicalConcept:
    """Represents a biological concept for CAV analysis."""

    concept_id: str
    name: str
    description: str
    category: str
    positive_examples: List[str]  # Gene symbols or sample IDs
    negative_examples: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConceptAnalysisResult:
    """Results from concept activation analysis."""

    concept: BiologicalConcept
    cav_vector: np.ndarray
    cav_accuracy: float
    tcav_scores: Dict[str, float]
    statistical_significance: Dict[str, float]
    directional_derivatives: np.ndarray
    concept_sensitivity: float
    examples_analysis: Dict[str, Any]


class ConceptActivationMapper:
    """
    Concept Activation Vector (CAV) mapper for biological interpretability.

    This class implements TCAV (Testing with Concept Activation Vectors) to understand
    what biological concepts a model has learned and how they influence predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        concept_library: Optional[Dict[str, BiologicalConcept]] = None,
    ):
        """
        Initialize CAV mapper.

        Args:
            model: The model to analyze
            layer_names: Names of layers to extract activations from
            concept_library: Dictionary of biological concepts
        """
        self.model = model
        self.layer_names = layer_names
        self.concept_library = concept_library or {}
        self.activation_hooks = {}
        self.activations = {}

        # Register hooks for activation extraction
        self._register_activation_hooks()

    def _register_activation_hooks(self):
        """Register forward hooks to extract activations."""

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
                else:
                    # Handle tuple/list outputs
                    self.activations[name] = (
                        output[0].detach() if isinstance(output, (tuple, list)) else output
                    )

            return hook

        # Find and register hooks for specified layers
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(get_activation(name))
                self.activation_hooks[name] = handle
                logger.info(f"Registered hook for layer: {name}")

    def extract_activations(
        self,
        data_loader: torch.utils.data.DataLoader,
        layer_name: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract activations from specified layer.

        Args:
            data_loader: DataLoader for input data
            layer_name: Name of layer to extract from
            max_samples: Maximum number of samples to process

        Returns:
            Tuple of (activations, sample_ids)
        """
        self.model.eval()
        all_activations = []
        sample_ids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if max_samples and len(all_activations) >= max_samples:
                    break

                # Forward pass
                if isinstance(batch, dict):
                    inputs = batch.get("image", batch.get("input", None))
                    if inputs is not None:
                        batch_sample_ids = batch.get(
                            "sample_id", [f"sample_{batch_idx}_{i}" for i in range(len(inputs))]
                        )
                    else:
                        batch_sample_ids = [f"sample_{batch_idx}_0"]
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    batch_sample_ids = [f"sample_{batch_idx}_{i}" for i in range(len(inputs))]

                # Move to device
                if inputs is not None and hasattr(self.model, "device"):
                    inputs = inputs.to(next(self.model.parameters()).device)

                # Forward pass to trigger hooks
                _ = self.model(inputs)

                # Extract activations
                if layer_name in self.activations:
                    activations = self.activations[layer_name]

                    # Flatten if needed (keep batch dimension)
                    if len(activations.shape) > 2:
                        activations = activations.view(activations.size(0), -1)

                    all_activations.append(activations.cpu().numpy())
                    sample_ids.extend(batch_sample_ids)

        if all_activations:
            return np.concatenate(all_activations, axis=0), sample_ids
        else:
            logger.warning(f"No activations extracted for layer: {layer_name}")
            return np.array([]), []

    def compute_cav(
        self, concept: BiologicalConcept, activations: np.ndarray, sample_ids: List[str]
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Concept Activation Vector (CAV) for a biological concept.

        Args:
            concept: Biological concept to compute CAV for
            activations: Activations from model layer
            sample_ids: Sample identifiers

        Returns:
            Tuple of (CAV vector, classification accuracy)
        """
        # Create labels based on concept examples
        labels = []
        concept_activations = []

        sample_id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}

        # Positive examples
        positive_indices = []
        for example in concept.positive_examples:
            if example in sample_id_to_idx:
                positive_indices.append(sample_id_to_idx[example])

        # Negative examples (random if not specified)
        if concept.negative_examples:
            negative_indices = []
            for example in concept.negative_examples:
                if example in sample_id_to_idx:
                    negative_indices.append(sample_id_to_idx[example])
        else:
            # Use random negative examples
            all_indices = set(range(len(sample_ids)))
            positive_set = set(positive_indices)
            negative_candidates = list(all_indices - positive_set)

            # Sample same number as positive examples
            n_negative = min(len(positive_indices) * 2, len(negative_candidates))
            negative_indices = np.random.choice(
                negative_candidates, n_negative, replace=False
            ).tolist()

        if not positive_indices or not negative_indices:
            logger.warning(f"Insufficient examples for concept: {concept.name}")
            return np.array([]), 0.0

        # Prepare training data
        selected_indices = positive_indices + negative_indices
        X = activations[selected_indices]
        y = [1] * len(positive_indices) + [0] * len(negative_indices)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train linear classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_scaled, y)

        # Get CAV (classifier weights)
        cav_vector = classifier.coef_[0]

        # Evaluate classifier accuracy
        y_pred = classifier.predict(X_scaled)
        accuracy = (y_pred == y).mean()

        logger.info(f"CAV computed for '{concept.name}': accuracy = {accuracy:.3f}")

        return cav_vector, accuracy

    def compute_tcav_scores(
        self,
        concept: BiologicalConcept,
        cav_vector: np.ndarray,
        test_activations: np.ndarray,
        test_labels: np.ndarray,
        target_class: int = 1,
    ) -> Dict[str, float]:
        """
        Compute TCAV (Testing with CAV) scores.

        Args:
            concept: Biological concept
            cav_vector: Concept activation vector
            test_activations: Test set activations
            test_labels: Test set labels
            target_class: Target class for analysis

        Returns:
            Dictionary of TCAV scores
        """
        # Filter for target class
        target_mask = test_labels == target_class
        if not target_mask.any():
            logger.warning(f"No samples found for target class {target_class}")
            return {}

        target_activations = test_activations[target_mask]

        # Compute directional derivatives (dot product with CAV)
        directional_derivatives = np.dot(target_activations, cav_vector)

        # TCAV score: fraction of positive directional derivatives
        tcav_score = (directional_derivatives > 0).mean()

        # Additional statistics
        tcav_scores = {
            "tcav_score": tcav_score,
            "mean_derivative": directional_derivatives.mean(),
            "std_derivative": directional_derivatives.std(),
            "positive_fraction": tcav_score,
            "negative_fraction": 1 - tcav_score,
            "effect_size": abs(directional_derivatives.mean()) / directional_derivatives.std()
            if directional_derivatives.std() > 0
            else 0,
        }

        return tcav_scores

    def statistical_significance_test(
        self,
        concept: BiologicalConcept,
        activations: np.ndarray,
        sample_ids: List[str],
        n_random_concepts: int = 50,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """
        Test statistical significance using random concept baseline.

        Args:
            concept: Biological concept to test
            activations: Layer activations
            sample_ids: Sample identifiers
            n_random_concepts: Number of random concepts for baseline
            alpha: Significance level

        Returns:
            Dictionary with significance test results
        """
        # Compute CAV for real concept
        real_cav, real_accuracy = self.compute_cav(concept, activations, sample_ids)

        if len(real_cav) == 0:
            return {"p_value": 1.0, "is_significant": False}

        # Generate random concepts and compute their CAVs
        random_accuracies = []

        for _ in range(n_random_concepts):
            # Create random concept with same number of examples
            n_positive = len(concept.positive_examples)
            n_negative = (
                len(concept.negative_examples) if concept.negative_examples else n_positive * 2
            )

            # Randomly sample examples
            available_samples = list(range(len(sample_ids)))
            random_positive = np.random.choice(available_samples, n_positive, replace=False)
            remaining_samples = [i for i in available_samples if i not in random_positive]
            random_negative = np.random.choice(
                remaining_samples, min(n_negative, len(remaining_samples)), replace=False
            )

            # Create random concept
            random_concept = BiologicalConcept(
                concept_id=f"random_{_}",
                name=f"Random Concept {_}",
                description="Random concept for significance testing",
                category="random",
                positive_examples=[sample_ids[i] for i in random_positive],
                negative_examples=[sample_ids[i] for i in random_negative],
            )

            # Compute CAV for random concept
            _, random_accuracy = self.compute_cav(random_concept, activations, sample_ids)
            random_accuracies.append(random_accuracy)

        # Compute p-value
        random_accuracies = np.array(random_accuracies)
        p_value = (random_accuracies >= real_accuracy).mean()

        return {
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "real_accuracy": real_accuracy,
            "random_mean_accuracy": random_accuracies.mean(),
            "random_std_accuracy": random_accuracies.std(),
            "z_score": (real_accuracy - random_accuracies.mean()) / random_accuracies.std()
            if random_accuracies.std() > 0
            else 0,
        }

    def analyze_concept(
        self,
        concept: BiologicalConcept,
        data_loader: torch.utils.data.DataLoader,
        layer_name: str,
        test_data_loader: Optional[torch.utils.data.DataLoader] = None,
        target_class: int = 1,
    ) -> ConceptAnalysisResult:
        """
        Perform complete concept analysis including CAV computation and TCAV scoring.

        Args:
            concept: Biological concept to analyze
            data_loader: DataLoader for training CAV
            layer_name: Layer to extract activations from
            test_data_loader: Optional separate test set
            target_class: Target class for TCAV analysis

        Returns:
            ConceptAnalysisResult object
        """
        logger.info(f"Analyzing concept: {concept.name}")

        # Extract activations
        activations, sample_ids = self.extract_activations(data_loader, layer_name)

        if len(activations) == 0:
            logger.error(f"No activations extracted for concept analysis: {concept.name}")
            return None

        # Compute CAV
        cav_vector, cav_accuracy = self.compute_cav(concept, activations, sample_ids)

        if len(cav_vector) == 0:
            logger.error(f"Failed to compute CAV for concept: {concept.name}")
            return None

        # Test statistical significance
        significance_results = self.statistical_significance_test(concept, activations, sample_ids)

        # TCAV analysis on test set
        tcav_scores = {}
        directional_derivatives = np.array([])

        if test_data_loader is not None:
            test_activations, test_sample_ids = self.extract_activations(
                test_data_loader, layer_name
            )

            if len(test_activations) > 0:
                # Create dummy labels for test set (assuming binary classification)
                test_labels = np.ones(len(test_activations))  # This should be actual labels

                tcav_scores = self.compute_tcav_scores(
                    concept, cav_vector, test_activations, test_labels, target_class
                )

                # Compute directional derivatives
                directional_derivatives = np.dot(test_activations, cav_vector)

        # Analyze examples
        examples_analysis = self._analyze_concept_examples(
            concept, activations, sample_ids, cav_vector
        )

        # Compute concept sensitivity
        concept_sensitivity = abs(cav_vector).mean()

        result = ConceptAnalysisResult(
            concept=concept,
            cav_vector=cav_vector,
            cav_accuracy=cav_accuracy,
            tcav_scores=tcav_scores,
            statistical_significance=significance_results,
            directional_derivatives=directional_derivatives,
            concept_sensitivity=concept_sensitivity,
            examples_analysis=examples_analysis,
        )

        logger.info(
            f"Concept analysis completed for '{concept.name}': "
            f"accuracy={cav_accuracy:.3f}, significance={'Yes' if significance_results['is_significant'] else 'No'}"
        )

        return result

    def _analyze_concept_examples(
        self,
        concept: BiologicalConcept,
        activations: np.ndarray,
        sample_ids: List[str],
        cav_vector: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze concept examples for interpretability."""
        sample_id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}

        # Analyze positive examples
        positive_analyses = {}
        for example in concept.positive_examples:
            if example in sample_id_to_idx:
                idx = sample_id_to_idx[example]
                activation = activations[idx]
                concept_activation = np.dot(activation, cav_vector)

                positive_analyses[example] = {
                    "concept_activation": concept_activation,
                    "activation_magnitude": np.linalg.norm(activation),
                    "top_features": np.argsort(activation)[-10:].tolist(),  # Top 10 features
                }

        # Analyze negative examples
        negative_analyses = {}
        if concept.negative_examples:
            for example in concept.negative_examples:
                if example in sample_id_to_idx:
                    idx = sample_id_to_idx[example]
                    activation = activations[idx]
                    concept_activation = np.dot(activation, cav_vector)

                    negative_analyses[example] = {
                        "concept_activation": concept_activation,
                        "activation_magnitude": np.linalg.norm(activation),
                        "top_features": np.argsort(activation)[-10:].tolist(),
                    }

        return {
            "positive_examples": positive_analyses,
            "negative_examples": negative_analyses,
            "concept_vector_stats": {
                "mean": cav_vector.mean(),
                "std": cav_vector.std(),
                "top_weights": np.argsort(np.abs(cav_vector))[-20:].tolist(),  # Top 20 weights
            },
        }

    def analyze_multiple_concepts(
        self,
        concepts: List[BiologicalConcept],
        data_loader: torch.utils.data.DataLoader,
        layer_names: Optional[List[str]] = None,
        test_data_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict[str, Dict[str, ConceptAnalysisResult]]:
        """
        Analyze multiple concepts across multiple layers.

        Args:
            concepts: List of biological concepts
            data_loader: DataLoader for training
            layer_names: List of layer names (uses all if None)
            test_data_loader: Optional test data loader

        Returns:
            Nested dictionary: {layer_name: {concept_id: ConceptAnalysisResult}}
        """
        if layer_names is None:
            layer_names = self.layer_names

        results = {}

        for layer_name in layer_names:
            logger.info(f"Analyzing concepts for layer: {layer_name}")
            results[layer_name] = {}

            for concept in concepts:
                try:
                    analysis_result = self.analyze_concept(
                        concept, data_loader, layer_name, test_data_loader
                    )
                    if analysis_result is not None:
                        results[layer_name][concept.concept_id] = analysis_result
                except Exception as e:
                    logger.error(
                        f"Failed to analyze concept {concept.name} in layer {layer_name}: {str(e)}"
                    )

        return results

    def cleanup(self):
        """Remove activation hooks."""
        for handle in self.activation_hooks.values():
            handle.remove()
        self.activation_hooks.clear()
        logger.info("Activation hooks removed")


class ConceptVisualizer:
    """Visualize concept activation analysis results."""

    def __init__(self):
        """Initialize visualizer."""
        self.color_palette = px.colors.qualitative.Set3

    def plot_concept_accuracies(
        self, results: Dict[str, Dict[str, ConceptAnalysisResult]], save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot CAV accuracies across concepts and layers."""
        # Prepare data
        data = []
        for layer_name, layer_results in results.items():
            for concept_id, result in layer_results.items():
                data.append(
                    {
                        "Layer": layer_name,
                        "Concept": result.concept.name,
                        "CAV_Accuracy": result.cav_accuracy,
                        "Significant": "Yes"
                        if result.statistical_significance["is_significant"]
                        else "No",
                        "P_Value": result.statistical_significance["p_value"],
                    }
                )

        if not data:
            return go.Figure()

        df = pd.DataFrame(data)

        # Create grouped bar plot
        fig = px.bar(
            df,
            x="Concept",
            y="CAV_Accuracy",
            color="Layer",
            facet_col="Significant",
            title="Concept Activation Vector Accuracies",
            hover_data=["P_Value"],
        )

        fig.update_layout(xaxis_tickangle=-45, height=500, template="plotly_white")

        # Add significance threshold line
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="red", annotation_text="Random baseline (0.5)"
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Concept accuracies plot saved to {save_path}")

        return fig

    def plot_tcav_scores(
        self, results: Dict[str, Dict[str, ConceptAnalysisResult]], save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot TCAV scores across concepts and layers."""
        # Prepare data
        data = []
        for layer_name, layer_results in results.items():
            for concept_id, result in layer_results.items():
                if result.tcav_scores:
                    data.append(
                        {
                            "Layer": layer_name,
                            "Concept": result.concept.name,
                            "TCAV_Score": result.tcav_scores.get("tcav_score", 0),
                            "Effect_Size": result.tcav_scores.get("effect_size", 0),
                            "Mean_Derivative": result.tcav_scores.get("mean_derivative", 0),
                        }
                    )

        if not data:
            return go.Figure()

        df = pd.DataFrame(data)

        # Create scatter plot
        fig = px.scatter(
            df,
            x="TCAV_Score",
            y="Effect_Size",
            color="Layer",
            size="Effect_Size",
            hover_name="Concept",
            hover_data=["Mean_Derivative"],
            title="TCAV Scores vs Effect Sizes",
        )

        # Add quadrant lines
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="TCAV = 0.5")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="TCAV Score (Concept Sensitivity)",
            yaxis_title="Effect Size",
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"TCAV scores plot saved to {save_path}")

        return fig

    def plot_concept_sensitivity_heatmap(
        self, results: Dict[str, Dict[str, ConceptAnalysisResult]], save_path: Optional[str] = None
    ) -> go.Figure:
        """Create heatmap of concept sensitivities across layers."""
        # Prepare data matrix
        concepts = set()
        layers = list(results.keys())

        for layer_results in results.values():
            concepts.update(result.concept.name for result in layer_results.values())

        concepts = sorted(list(concepts))

        # Create sensitivity matrix
        sensitivity_matrix = np.zeros((len(concepts), len(layers)))

        for layer_idx, layer_name in enumerate(layers):
            layer_results = results[layer_name]
            for concept_idx, concept_name in enumerate(concepts):
                # Find matching result
                for result in layer_results.values():
                    if result.concept.name == concept_name:
                        sensitivity_matrix[concept_idx, layer_idx] = result.concept_sensitivity
                        break

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=sensitivity_matrix,
                x=layers,
                y=concepts,
                colorscale="RdYlBu_r",
                hovertemplate="Layer: %{x}<br>Concept: %{y}<br>Sensitivity: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Concept Sensitivity Across Layers",
            xaxis_title="Layer",
            yaxis_title="Biological Concept",
            height=max(400, len(concepts) * 25),
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Concept sensitivity heatmap saved to {save_path}")

        return fig

    def plot_directional_derivatives(
        self, result: ConceptAnalysisResult, save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot distribution of directional derivatives for a concept."""
        if len(result.directional_derivatives) == 0:
            return go.Figure()

        derivatives = result.directional_derivatives

        # Create histogram
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=derivatives,
                nbinsx=50,
                name="Directional Derivatives",
                opacity=0.7,
                marker_color="lightblue",
            )
        )

        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero derivative")

        # Add mean line
        mean_derivative = derivatives.mean()
        fig.add_vline(
            x=mean_derivative,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Mean: {mean_derivative:.3f}",
        )

        fig.update_layout(
            title=f"Directional Derivatives for {result.concept.name}",
            xaxis_title="Directional Derivative",
            yaxis_title="Count",
            template="plotly_white",
            height=400,
            annotations=[
                dict(
                    x=0.7,
                    y=0.9,
                    xref="paper",
                    yref="paper",
                    text=f"TCAV Score: {result.tcav_scores.get('tcav_score', 0):.3f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                )
            ],
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Directional derivatives plot saved to {save_path}")

        return fig

    def create_concept_dashboard(
        self, results: Dict[str, Dict[str, ConceptAnalysisResult]], save_path: str
    ) -> None:
        """Create comprehensive concept analysis dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "CAV Accuracies by Concept",
                "TCAV Scores vs Effect Sizes",
                "Concept Sensitivity Heatmap",
                "Significance Test Results",
                "Concept Categories Distribution",
                "Layer-wise Analysis Summary",
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "table"}],
            ],
        )

        # Prepare data
        concept_data = []
        for layer_name, layer_results in results.items():
            for result in layer_results.values():
                concept_data.append(
                    {
                        "layer": layer_name,
                        "concept": result.concept.name,
                        "category": result.concept.category,
                        "cav_accuracy": result.cav_accuracy,
                        "tcav_score": result.tcav_scores.get("tcav_score", 0),
                        "effect_size": result.tcav_scores.get("effect_size", 0),
                        "p_value": result.statistical_significance["p_value"],
                        "is_significant": result.statistical_significance["is_significant"],
                        "sensitivity": result.concept_sensitivity,
                    }
                )

        if not concept_data:
            logger.warning("No concept data available for dashboard")
            return

        df = pd.DataFrame(concept_data)

        # 1. CAV Accuracies
        accuracy_by_concept = (
            df.groupby("concept")["cav_accuracy"].mean().sort_values(ascending=False)
        )
        fig.add_trace(
            go.Bar(x=accuracy_by_concept.index, y=accuracy_by_concept.values, name="CAV Accuracy"),
            row=1,
            col=1,
        )

        # 2. TCAV Scores vs Effect Sizes
        fig.add_trace(
            go.Scatter(
                x=df["tcav_score"],
                y=df["effect_size"],
                mode="markers",
                text=df["concept"],
                name="TCAV Analysis",
                marker=dict(size=8, color=df["sensitivity"], colorscale="Viridis"),
            ),
            row=1,
            col=2,
        )

        # 3. Concept Sensitivity Heatmap (simplified)
        pivot_table = df.pivot_table(
            values="sensitivity", index="concept", columns="layer", fill_value=0
        )
        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale="RdYlBu_r",
                showscale=False,
            ),
            row=2,
            col=1,
        )

        # 4. Category Distribution
        category_counts = df["category"].value_counts()
        fig.add_trace(
            go.Bar(x=category_counts.index, y=category_counts.values, name="Categories"),
            row=3,
            col=1,
        )

        # 5. Summary Table
        summary_data = (
            df.groupby("layer")
            .agg(
                {
                    "cav_accuracy": "mean",
                    "tcav_score": "mean",
                    "is_significant": "sum",
                    "sensitivity": "mean",
                }
            )
            .round(3)
        )

        fig.add_trace(
            go.Table(
                header=dict(values=["Layer"] + list(summary_data.columns)),
                cells=dict(
                    values=[summary_data.index]
                    + [summary_data[col] for col in summary_data.columns]
                ),
            ),
            row=3,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Biological Concept Activation Analysis Dashboard",
            height=1200,
            template="plotly_white",
            showlegend=False,
        )

        # Save dashboard
        fig.write_html(save_path)
        logger.info(f"Concept analysis dashboard saved to {save_path}")


def discover_biological_concepts(
    gene_expression_data: pd.DataFrame,
    pathway_database: Optional[Dict] = None,
    min_genes_per_concept: int = 10,
    max_genes_per_concept: int = 200,
) -> List[BiologicalConcept]:
    """
    Automatically discover biological concepts from gene expression data.

    Args:
        gene_expression_data: DataFrame with genes as columns, samples as rows
        pathway_database: Optional pathway database for concept discovery
        min_genes_per_concept: Minimum number of genes per concept
        max_genes_per_concept: Maximum number of genes per concept

    Returns:
        List of discovered biological concepts
    """
    concepts = []

    # Method 1: Use pathway database if available
    if pathway_database:
        for pathway_id, pathway_info in pathway_database.items():
            genes = pathway_info.get("genes", [])

            # Filter genes present in data
            available_genes = [g for g in genes if g in gene_expression_data.columns]

            if min_genes_per_concept <= len(available_genes) <= max_genes_per_concept:
                concept = BiologicalConcept(
                    concept_id=pathway_id,
                    name=pathway_info.get("name", pathway_id),
                    description=pathway_info.get("description", ""),
                    category=pathway_info.get("category", "pathway"),
                    positive_examples=available_genes,
                    metadata={"source": "pathway_database", "original_size": len(genes)},
                )
                concepts.append(concept)

    # Method 2: Discover concepts from gene co-expression
    logger.info("Discovering concepts from gene co-expression...")

    # Compute gene correlation matrix
    gene_corr = gene_expression_data.corr()

    # Find highly correlated gene clusters
    from sklearn.cluster import SpectralClustering

    # Use spectral clustering on correlation matrix
    n_clusters = min(
        50, len(gene_expression_data.columns) // 20
    )  # Heuristic for number of clusters

    if n_clusters > 1:
        clustering = SpectralClustering(
            n_clusters=n_clusters, random_state=42, affinity="precomputed"
        )

        # Convert correlation to similarity matrix
        similarity_matrix = (gene_corr + 1) / 2  # Normalize to [0, 1]
        similarity_matrix = np.maximum(similarity_matrix, 0)  # Ensure non-negative

        try:
            cluster_labels = clustering.fit_predict(similarity_matrix)

            # Create concepts from clusters
            for cluster_id in range(n_clusters):
                cluster_genes = gene_expression_data.columns[cluster_labels == cluster_id].tolist()

                if min_genes_per_concept <= len(cluster_genes) <= max_genes_per_concept:
                    concept = BiologicalConcept(
                        concept_id=f"coexpr_cluster_{cluster_id}",
                        name=f"Co-expression Cluster {cluster_id}",
                        description=f"Genes with similar expression patterns (cluster {cluster_id})",
                        category="co-expression",
                        positive_examples=cluster_genes,
                        metadata={"source": "co-expression", "cluster_id": cluster_id},
                    )
                    concepts.append(concept)

        except Exception as e:
            logger.warning(f"Co-expression clustering failed: {str(e)}")

    # Method 3: Create concepts based on expression variance
    logger.info("Creating variance-based concepts...")

    # High variance genes (highly variable)
    gene_vars = gene_expression_data.var()
    high_var_genes = gene_vars[gene_vars > gene_vars.quantile(0.8)].index.tolist()

    if len(high_var_genes) >= min_genes_per_concept:
        concepts.append(
            BiologicalConcept(
                concept_id="high_variance_genes",
                name="Highly Variable Genes",
                description="Genes with high expression variance across samples",
                category="expression_pattern",
                positive_examples=high_var_genes[:max_genes_per_concept],
                metadata={"source": "variance_analysis", "threshold": gene_vars.quantile(0.8)},
            )
        )

    # Low variance genes (stably expressed)
    low_var_genes = gene_vars[gene_vars < gene_vars.quantile(0.2)].index.tolist()

    if len(low_var_genes) >= min_genes_per_concept:
        concepts.append(
            BiologicalConcept(
                concept_id="low_variance_genes",
                name="Stably Expressed Genes",
                description="Genes with low expression variance across samples",
                category="expression_pattern",
                positive_examples=low_var_genes[:max_genes_per_concept],
                metadata={"source": "variance_analysis", "threshold": gene_vars.quantile(0.2)},
            )
        )

    logger.info(f"Discovered {len(concepts)} biological concepts")

    return concepts


def compute_concept_activations(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    concepts: List[BiologicalConcept],
    layer_names: List[str],
    output_dir: str = "concept_analysis_results",
) -> Dict[str, Any]:
    """
    Compute concept activations for a list of biological concepts.

    Args:
        model: Model to analyze
        data_loader: Data loader for analysis
        concepts: List of biological concepts
        layer_names: Names of layers to analyze
        output_dir: Output directory for results

    Returns:
        Dict containing analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize concept mapper
    concept_mapper = ConceptActivationMapper(
        model, layer_names, {c.concept_id: c for c in concepts}
    )

    try:
        logger.info(
            f"Starting concept activation analysis for {len(concepts)} concepts across {len(layer_names)} layers"
        )

        # Analyze all concepts
        all_results = concept_mapper.analyze_multiple_concepts(
            concepts=concepts, data_loader=data_loader, layer_names=layer_names
        )

        # Create visualizations
        visualizer = ConceptVisualizer()

        # Generate plots
        logger.info("Generating visualizations...")

        # CAV accuracies plot
        accuracy_fig = visualizer.plot_concept_accuracies(
            all_results, save_path=str(output_path / "concept_accuracies.html")
        )

        # TCAV scores plot
        tcav_fig = visualizer.plot_tcav_scores(
            all_results, save_path=str(output_path / "tcav_scores.html")
        )

        # Sensitivity heatmap
        sensitivity_fig = visualizer.plot_concept_sensitivity_heatmap(
            all_results, save_path=str(output_path / "concept_sensitivity_heatmap.html")
        )

        # Create comprehensive dashboard
        visualizer.create_concept_dashboard(
            all_results, save_path=str(output_path / "concept_analysis_dashboard.html")
        )

        # Generate individual concept plots
        individual_plots_dir = output_path / "individual_concepts"
        individual_plots_dir.mkdir(exist_ok=True)

        for layer_name, layer_results in all_results.items():
            for concept_id, result in layer_results.items():
                if len(result.directional_derivatives) > 0:
                    derivative_fig = visualizer.plot_directional_derivatives(
                        result,
                        save_path=str(
                            individual_plots_dir / f"{concept_id}_{layer_name}_derivatives.html"
                        ),
                    )

        # Save detailed results
        detailed_results = {
            "analysis_results": {},
            "summary_statistics": {},
            "concept_metadata": {
                c.concept_id: {
                    "name": c.name,
                    "description": c.description,
                    "category": c.category,
                    "num_positive_examples": len(c.positive_examples),
                    "num_negative_examples": len(c.negative_examples) if c.negative_examples else 0,
                }
                for c in concepts
            },
        }

        # Process results for JSON serialization
        for layer_name, layer_results in all_results.items():
            detailed_results["analysis_results"][layer_name] = {}

            layer_stats = {
                "total_concepts": len(layer_results),
                "significant_concepts": 0,
                "mean_cav_accuracy": 0,
                "mean_tcav_score": 0,
                "mean_sensitivity": 0,
            }

            accuracies = []
            tcav_scores = []
            sensitivities = []

            for concept_id, result in layer_results.items():
                # Convert result to serializable format
                result_dict = {
                    "concept_id": result.concept.concept_id,
                    "concept_name": result.concept.name,
                    "cav_accuracy": float(result.cav_accuracy),
                    "concept_sensitivity": float(result.concept_sensitivity),
                    "statistical_significance": {
                        k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in result.statistical_significance.items()
                    },
                    "tcav_scores": {
                        k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in result.tcav_scores.items()
                    }
                    if result.tcav_scores
                    else {},
                    "examples_analysis": {
                        "num_positive_analyzed": len(result.examples_analysis["positive_examples"]),
                        "num_negative_analyzed": len(result.examples_analysis["negative_examples"]),
                        "concept_vector_mean": float(
                            result.examples_analysis["concept_vector_stats"]["mean"]
                        ),
                        "concept_vector_std": float(
                            result.examples_analysis["concept_vector_stats"]["std"]
                        ),
                    },
                }

                detailed_results["analysis_results"][layer_name][concept_id] = result_dict

                # Accumulate statistics
                accuracies.append(result.cav_accuracy)
                sensitivities.append(result.concept_sensitivity)

                if result.statistical_significance["is_significant"]:
                    layer_stats["significant_concepts"] += 1

                if result.tcav_scores and "tcav_score" in result.tcav_scores:
                    tcav_scores.append(result.tcav_scores["tcav_score"])

            # Compute layer statistics
            if accuracies:
                layer_stats["mean_cav_accuracy"] = float(np.mean(accuracies))
                layer_stats["mean_sensitivity"] = float(np.mean(sensitivities))

            if tcav_scores:
                layer_stats["mean_tcav_score"] = float(np.mean(tcav_scores))

            detailed_results["summary_statistics"][layer_name] = layer_stats

        # Save results to JSON
        with open(output_path / "concept_analysis_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)

        # Generate summary report
        generate_concept_analysis_report(
            detailed_results, str(output_path / "concept_analysis_report.md")
        )

        logger.info(f"Concept activation analysis completed. Results saved to: {output_path}")

        return {
            "detailed_results": detailed_results,
            "analysis_results": all_results,
            "output_directory": str(output_path),
            "visualizations": {
                "dashboard": str(output_path / "concept_analysis_dashboard.html"),
                "accuracies": str(output_path / "concept_accuracies.html"),
                "tcav_scores": str(output_path / "tcav_scores.html"),
                "sensitivity_heatmap": str(output_path / "concept_sensitivity_heatmap.html"),
            },
        }

    finally:
        # Clean up hooks
        concept_mapper.cleanup()


def generate_concept_analysis_report(results: Dict[str, Any], output_file: str):
    """Generate a comprehensive concept analysis report."""
    report_lines = [
        "# Biological Concept Activation Analysis Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
    ]

    # Overall statistics
    total_concepts = sum(
        stats["total_concepts"] for stats in results["summary_statistics"].values()
    )
    total_significant = sum(
        stats["significant_concepts"] for stats in results["summary_statistics"].values()
    )

    report_lines.extend(
        [
            f"- **Total concepts analyzed:** {total_concepts}",
            f"- **Statistically significant concepts:** {total_significant} ({total_significant/total_concepts*100:.1f}%)"
            if total_concepts > 0
            else "- **Statistically significant concepts:** 0",
            f"- **Layers analyzed:** {len(results['summary_statistics'])}",
            "",
            "## Layer-wise Analysis",
            "",
        ]
    )

    # Layer statistics
    for layer_name, stats in results["summary_statistics"].items():
        report_lines.extend(
            [
                f"### {layer_name}",
                "",
                f"- **Concepts analyzed:** {stats['total_concepts']}",
                f"- **Significant concepts:** {stats['significant_concepts']} ({stats['significant_concepts']/stats['total_concepts']*100:.1f}%)"
                if stats["total_concepts"] > 0
                else "- **Significant concepts:** 0",
                f"- **Mean CAV accuracy:** {stats['mean_cav_accuracy']:.3f}",
                f"- **Mean concept sensitivity:** {stats['mean_sensitivity']:.3f}",
            ]
        )

        if stats["mean_tcav_score"] > 0:
            report_lines.append(f"- **Mean TCAV score:** {stats['mean_tcav_score']:.3f}")

        report_lines.append("")

    # Concept categories
    category_counts = defaultdict(int)
    for concept_info in results["concept_metadata"].values():
        category_counts[concept_info["category"]] += 1

    if category_counts:
        report_lines.extend(
            [
                "## Concept Categories",
                "",
            ]
        )

        for category, count in sorted(category_counts.items()):
            report_lines.append(f"- **{category.title()}:** {count} concepts")

        report_lines.append("")

    # Top performing concepts
    report_lines.extend(["## Top Performing Concepts", "", "### Highest CAV Accuracies", ""])

    # Collect all concept results for ranking
    all_concept_results = []
    for layer_name, layer_results in results["analysis_results"].items():
        for concept_id, result in layer_results.items():
            all_concept_results.append(
                {
                    "layer": layer_name,
                    "concept_id": concept_id,
                    "concept_name": result["concept_name"],
                    "cav_accuracy": result["cav_accuracy"],
                    "is_significant": result["statistical_significance"]["is_significant"],
                    "tcav_score": result["tcav_scores"].get("tcav_score", 0)
                    if result["tcav_scores"]
                    else 0,
                    "sensitivity": result["concept_sensitivity"],
                }
            )

    # Sort by CAV accuracy
    top_accuracy = sorted(all_concept_results, key=lambda x: x["cav_accuracy"], reverse=True)[:10]

    for i, result in enumerate(top_accuracy, 1):
        significance = "" if result["is_significant"] else ""
        report_lines.append(
            f"{i}. **{result['concept_name']}** ({result['layer']}) - Accuracy: {result['cav_accuracy']:.3f} {significance}"
        )

    # Significant concepts
    significant_concepts = [r for r in all_concept_results if r["is_significant"]]

    if significant_concepts:
        report_lines.extend(["", "### Statistically Significant Concepts", ""])

        for i, result in enumerate(significant_concepts[:15], 1):
            report_lines.append(
                f"{i}. **{result['concept_name']}** ({result['layer']}) - Accuracy: {result['cav_accuracy']:.3f}"
            )

    # Recommendations
    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )

    if total_significant / total_concepts > 0.3 if total_concepts > 0 else False:
        report_lines.append(
            "SUCCESS: **High concept significance rate** - Model shows good biological concept learning"
        )
    else:
        report_lines.append(
            "WARNING: **Low concept significance rate** - Consider model architecture or training improvements"
        )

    if len([r for r in all_concept_results if r["cav_accuracy"] > 0.7]) > 5:
        report_lines.append(
            "SUCCESS: **Strong concept separability** - Model effectively distinguishes biological concepts"
        )
    else:
        report_lines.append(
            "NOTE: **Improve concept separability** - Consider feature engineering or data augmentation"
        )

    # Layer-specific recommendations
    best_layer = max(results["summary_statistics"].items(), key=lambda x: x[1]["mean_cav_accuracy"])
    report_lines.append(
        f"TARGET: **Best performing layer: {best_layer[0]}** - Focus analysis on this layer"
    )

    report_lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "1. **Focus on significant concepts** for biological interpretation",
            "2. **Investigate high-accuracy concepts** for potential biomarkers",
            "3. **Validate findings** with domain experts and literature",
            "4. **Use concept insights** for model improvement and feature selection",
            "",
            "---",
            "",
            "*This report was generated by OpenPerturbations Concept Activation Analysis*",
        ]
    )

    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Concept analysis report saved to {output_file}")


def create_biological_concept_library() -> List[BiologicalConcept]:
    """Create a standard library of biological concepts for analysis."""
    concepts = []

    # Cell cycle concepts
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="cell_cycle_g1s",
                name="G1/S Transition",
                description="Genes involved in G1 to S phase transition",
                category="cell_cycle",
                positive_examples=[
                    "CDK2",
                    "CCNE1",
                    "CCNE2",
                    "CDC25A",
                    "RB1",
                    "E2F1",
                    "E2F2",
                    "PCNA",
                ],
            ),
            BiologicalConcept(
                concept_id="cell_cycle_g2m",
                name="G2/M Transition",
                description="Genes involved in G2 to M phase transition",
                category="cell_cycle",
                positive_examples=[
                    "CDK1",
                    "CCNB1",
                    "CCNB2",
                    "CDC25B",
                    "CDC25C",
                    "PLK1",
                    "AURKA",
                    "AURKB",
                ],
            ),
            BiologicalConcept(
                concept_id="cell_cycle_checkpoints",
                name="Cell Cycle Checkpoints",
                description="DNA damage checkpoint and spindle checkpoint genes",
                category="cell_cycle",
                positive_examples=[
                    "TP53",
                    "ATM",
                    "ATR",
                    "CHEK1",
                    "CHEK2",
                    "BUB1",
                    "BUB3",
                    "MAD2L1",
                ],
            ),
        ]
    )

    # Apoptosis concepts
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="apoptosis_intrinsic",
                name="Intrinsic Apoptosis",
                description="Mitochondrial apoptosis pathway genes",
                category="apoptosis",
                positive_examples=[
                    "BAX",
                    "BAK1",
                    "BCL2",
                    "BCL2L1",
                    "APAF1",
                    "CYCS",
                    "CASP9",
                    "CASP3",
                ],
            ),
            BiologicalConcept(
                concept_id="apoptosis_extrinsic",
                name="Extrinsic Apoptosis",
                description="Death receptor apoptosis pathway genes",
                category="apoptosis",
                positive_examples=[
                    "FAS",
                    "FASL",
                    "TNF",
                    "TNFRSF1A",
                    "TRADD",
                    "FADD",
                    "CASP8",
                    "CASP10",
                ],
            ),
        ]
    )

    # DNA repair concepts
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="dna_repair_homologous",
                name="Homologous Recombination",
                description="Homologous recombination DNA repair genes",
                category="dna_repair",
                positive_examples=[
                    "BRCA1",
                    "BRCA2",
                    "RAD51",
                    "RAD52",
                    "RAD54L",
                    "PALB2",
                    "ATM",
                    "NBN",
                ],
            ),
            BiologicalConcept(
                concept_id="dna_repair_nhej",
                name="Non-Homologous End Joining",
                description="NHEJ DNA repair pathway genes",
                category="dna_repair",
                positive_examples=["XRCC6", "XRCC5", "PRKDC", "LIG4", "XRCC4", "NHEJ1", "DCLRE1C"],
            ),
            BiologicalConcept(
                concept_id="dna_repair_mismatch",
                name="Mismatch Repair",
                description="DNA mismatch repair genes",
                category="dna_repair",
                positive_examples=["MLH1", "MSH2", "MSH6", "PMS2", "MSH3", "MLH3", "PMS1"],
            ),
        ]
    )

    # Metabolism concepts
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="metabolism_glycolysis",
                name="Glycolysis",
                description="Glycolytic pathway enzymes",
                category="metabolism",
                positive_examples=[
                    "HK1",
                    "HK2",
                    "GPI",
                    "PFKL",
                    "PFKM",
                    "ALDOA",
                    "TPI1",
                    "GAPDH",
                    "PGK1",
                    "PGAM1",
                    "ENO1",
                    "PKM",
                ],
            ),
            BiologicalConcept(
                concept_id="metabolism_oxidative_phosphorylation",
                name="Oxidative Phosphorylation",
                description="Mitochondrial electron transport chain genes",
                category="metabolism",
                positive_examples=[
                    "NDUFA1",
                    "NDUFB1",
                    "SDHB",
                    "UQCR1",
                    "COX4I1",
                    "ATP5A1",
                    "ATP5B",
                    "CYC1",
                ],
            ),
            BiologicalConcept(
                concept_id="metabolism_fatty_acid",
                name="Fatty Acid Metabolism",
                description="Fatty acid synthesis and oxidation genes",
                category="metabolism",
                positive_examples=[
                    "FASN",
                    "ACACA",
                    "ACACB",
                    "CPT1A",
                    "CPT2",
                    "ACADVL",
                    "HADHA",
                    "HADHB",
                ],
            ),
        ]
    )

    # Signal transduction concepts
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="signaling_mapk",
                name="MAPK Signaling",
                description="MAP kinase signaling pathway genes",
                category="signaling",
                positive_examples=[
                    "KRAS",
                    "BRAF",
                    "MAP2K1",
                    "MAP2K2",
                    "MAPK1",
                    "MAPK3",
                    "ELK1",
                    "JUN",
                    "FOS",
                ],
            ),
            BiologicalConcept(
                concept_id="signaling_pi3k_akt",
                name="PI3K/AKT Signaling",
                description="PI3K/AKT pathway genes",
                category="signaling",
                positive_examples=[
                    "PIK3CA",
                    "PIK3CB",
                    "AKT1",
                    "AKT2",
                    "PTEN",
                    "TSC1",
                    "TSC2",
                    "MTOR",
                    "RICTOR",
                ],
            ),
            BiologicalConcept(
                concept_id="signaling_p53",
                name="p53 Signaling",
                description="p53 tumor suppressor pathway genes",
                category="signaling",
                positive_examples=[
                    "TP53",
                    "MDM2",
                    "CDKN1A",
                    "BBC3",
                    "BAX",
                    "GADD45A",
                    "SESN1",
                    "RRM2B",
                ],
            ),
        ]
    )

    # Transcription and epigenetics
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="transcription_chromatin_remodeling",
                name="Chromatin Remodeling",
                description="Chromatin remodeling complex genes",
                category="transcription",
                positive_examples=[
                    "SMARCA4",
                    "SMARCC1",
                    "ARID1A",
                    "ARID1B",
                    "CHD1",
                    "CHD4",
                    "HDAC1",
                    "HDAC2",
                ],
            ),
            BiologicalConcept(
                concept_id="transcription_histone_modification",
                name="Histone Modification",
                description="Histone methyltransferases and demethylases",
                category="transcription",
                positive_examples=[
                    "KMT2A",
                    "KMT2D",
                    "EZH2",
                    "SUZ12",
                    "KDM1A",
                    "KDM4A",
                    "KDM6A",
                    "KDM6B",
                ],
            ),
        ]
    )

    # Cancer hallmarks
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="cancer_proliferation",
                name="Sustained Proliferation",
                description="Genes promoting continuous cell proliferation",
                category="cancer_hallmarks",
                positive_examples=["MYC", "CCND1", "CDK4", "CDK6", "RB1", "E2F1", "TERT", "hTERT"],
            ),
            BiologicalConcept(
                concept_id="cancer_angiogenesis",
                name="Angiogenesis",
                description="Blood vessel formation genes",
                category="cancer_hallmarks",
                positive_examples=[
                    "VEGFA",
                    "VEGFR2",
                    "ANG",
                    "ANGPT1",
                    "ANGPT2",
                    "TIE1",
                    "FLT1",
                    "KDR",
                ],
            ),
            BiologicalConcept(
                concept_id="cancer_invasion_metastasis",
                name="Invasion and Metastasis",
                description="Genes involved in cancer invasion and metastasis",
                category="cancer_hallmarks",
                positive_examples=[
                    "CDH1",
                    "CDH2",
                    "VIM",
                    "SNAI1",
                    "SNAI2",
                    "TWIST1",
                    "ZEB1",
                    "MMP2",
                    "MMP9",
                ],
            ),
        ]
    )

    # Stress response
    concepts.extend(
        [
            BiologicalConcept(
                concept_id="stress_heat_shock",
                name="Heat Shock Response",
                description="Heat shock protein genes",
                category="stress_response",
                positive_examples=[
                    "HSP90AA1",
                    "HSP90AB1",
                    "HSPA1A",
                    "HSPA1B",
                    "HSPA4",
                    "HSPA8",
                    "HSPD1",
                    "HSPE1",
                ],
            ),
            BiologicalConcept(
                concept_id="stress_oxidative",
                name="Oxidative Stress Response",
                description="Antioxidant and oxidative stress response genes",
                category="stress_response",
                positive_examples=["SOD1", "SOD2", "CAT", "GPX1", "GSR", "GCLC", "NRF2", "KEAP1"],
            ),
        ]
    )

    logger.info(
        f"Created biological concept library with {len(concepts)} concepts across {len(set(c.category for c in concepts))} categories"
    )

    return concepts


def load_concept_library_from_file(file_path: str) -> List[BiologicalConcept]:
    """Load biological concepts from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    concepts = []
    for concept_data in data:
        concept = BiologicalConcept(
            concept_id=concept_data["concept_id"],
            name=concept_data["name"],
            description=concept_data["description"],
            category=concept_data["category"],
            positive_examples=concept_data["positive_examples"],
        )
