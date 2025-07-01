"""
Causal Intervention Analysis Module

This module provides tools for analyzing and predicting the effects of causal interventions
in biological systems, with specific focus on perturbation biology applications.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import networkx as nx
from dataclasses import dataclass
from abc import ABC, abstractmethod
# Lazy import scikit-learn metrics to avoid SciPy dependency in constrained environments
try:
    from sklearn.metrics import accuracy_score as _sk_accuracy, mean_squared_error as _sk_mse
    SKLEARN_METRICS_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False
from numpy.typing import NDArray

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class InterventionResult:
    """Result of a causal intervention analysis."""
    intervention_type: str
    target_variables: List[int]
    effect_magnitude: float
    effect_direction: str  # 'positive', 'negative', 'mixed'
    confidence: float
    affected_variables: List[int]
    downstream_effects: Dict[int, float]
    counterfactual_outcomes: Optional[np.ndarray] = None
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    biological_interpretation: Optional[str] = None
    experimental_feasibility: Optional[float] = None

@dataclass
class InterventionDesign:
    """Design specification for a causal intervention."""
    intervention_id: str
    intervention_type: str  # 'knockout', 'overexpression', 'drug_treatment', 'environmental'
    target_variables: List[int]
    intervention_strength: float
    duration: Optional[float] = None
    dose_response: Optional[List[float]] = None
    control_conditions: Optional[List[str]] = None
    expected_outcomes: Optional[Dict[str, float]] = None
    cost_estimate: Optional[float] = None
    timeline_estimate: Optional[float] = None

class InterventionPredictor(ABC):
    """Abstract base class for intervention effect prediction."""
    
    @abstractmethod
    def predict_intervention_effect(self, 
                                  intervention: InterventionDesign,
                                  baseline_data: np.ndarray) -> InterventionResult:
        """Predict the effect of an intervention."""
        pass
    
    @abstractmethod
    def compute_confidence(self, 
                          intervention: InterventionDesign,
                          baseline_data: np.ndarray) -> float:
        """Compute confidence in intervention prediction."""
        pass

    @abstractmethod
    def update_with_data(self, data: np.ndarray):
        """Update the predictor with new data."""
        pass

class CausalGraphInterventionPredictor(InterventionPredictor):
    """
    Intervention effect prediction using causal graph structure.
    
    Uses do-calculus and graph-based inference to predict intervention effects.
    """
    
    def __init__(self, 
                 causal_graph: np.ndarray,
                 variable_names: Optional[List[str]] = None,
                 confidence_scores: Optional[np.ndarray] = None):
        """
        Initialize with causal graph structure.
        
        Args:
            causal_graph: Adjacency matrix of causal relationships
            variable_names: Names of variables (optional)
            confidence_scores: Confidence in each edge (optional)
        """
        self.causal_graph = causal_graph
        self.variable_names = variable_names or [f"Var_{i}" for i in range(causal_graph.shape[0])]
        self.confidence_scores = confidence_scores
        self.nx_graph = nx.from_numpy_array(causal_graph, create_using=nx.DiGraph)
        
        # Precompute graph properties
        self._precompute_graph_properties()
    
    def _precompute_graph_properties(self):
        """Precompute useful graph properties for faster inference."""
        self.ancestors = {}
        self.descendants = {}
        self.markov_blankets = {}
        
        for node in self.nx_graph.nodes():
            self.ancestors[node] = set(nx.ancestors(self.nx_graph, node))
            self.descendants[node] = set(nx.descendants(self.nx_graph, node))
            
            # Markov blanket: parents + children + parents of children
            parents = set(self.nx_graph.predecessors(node))
            children = set(self.nx_graph.successors(node))
            parents_of_children = set()
            for child in children:
                parents_of_children.update(self.nx_graph.predecessors(child))
            
            self.markov_blankets[node] = parents | children | parents_of_children
            if node in self.markov_blankets[node]:
                self.markov_blankets[node].remove(node)
    
    def update_with_data(self, data: np.ndarray):
        """Update the predictor with new data (not applicable for static graph)."""
        logger.info("CausalGraphInterventionPredictor is static; not updated with new data.")
        pass
    
    def predict_intervention_effect(self, 
                                  intervention: InterventionDesign,
                                  baseline_data: np.ndarray) -> InterventionResult:
        """Predict intervention effects using causal graph inference."""
        
        logger.info(f"Predicting effects of intervention: {intervention.intervention_id}")
        
        # Identify affected variables
        affected_vars = self._identify_affected_variables(intervention.target_variables)
        
        # Compute downstream effects
        downstream_effects = self._compute_downstream_effects(
            intervention.target_variables, 
            intervention.intervention_strength
        )
        
        # Estimate effect magnitude and direction
        effect_magnitude = self._estimate_effect_magnitude(downstream_effects)
        effect_direction = self._determine_effect_direction(downstream_effects)
        
        # Compute confidence
        confidence = self.compute_confidence(intervention, baseline_data)
        
        # Generate counterfactual outcomes if baseline data provided
        counterfactual_outcomes = None
        if baseline_data is not None:
            counterfactual_outcomes = self._generate_counterfactual_outcomes(
                intervention, baseline_data
            )
        
        # Compute uncertainty bounds
        uncertainty_bounds = self._compute_uncertainty_bounds(
            downstream_effects, confidence
        )
        
        # Generate biological interpretation
        biological_interpretation = self._generate_biological_interpretation(
            intervention, downstream_effects
        )
        
        return InterventionResult(
            intervention_type=intervention.intervention_type,
            target_variables=intervention.target_variables,
            effect_magnitude=effect_magnitude,
            effect_direction=effect_direction,
            confidence=confidence,
            affected_variables=list(affected_vars),
            downstream_effects=downstream_effects,
            counterfactual_outcomes=counterfactual_outcomes,
            uncertainty_bounds=uncertainty_bounds,
            biological_interpretation=biological_interpretation,
            experimental_feasibility=self._estimate_experimental_feasibility(intervention)
        )
    
    def _identify_affected_variables(self, target_variables: List[int]) -> set:
        """Identify all variables affected by intervention on targets."""
        affected = set(target_variables)
        
        # Add all descendants of target variables
        for target in target_variables:
            if target in self.descendants:
                affected.update(self.descendants[target])
        
        return affected
    
    def _compute_downstream_effects(self, 
                                  target_variables: List[int], 
                                  intervention_strength: float) -> Dict[int, float]:
        """Compute downstream effects of intervention."""
        downstream_effects = {}
        
        for target in target_variables:
            # Direct effect on target
            downstream_effects[target] = intervention_strength
            
            # Propagate effects through causal paths
            if target in self.descendants:
                for descendant in self.descendants[target]:
                    # Compute effect through all paths
                    total_effect = self._compute_total_causal_effect(target, descendant)
                    downstream_effects[descendant] = intervention_strength * total_effect
        
        return downstream_effects
    
    def _compute_total_causal_effect(self, source: int, target: int) -> float:
        """Compute total causal effect from source to target."""
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(self.nx_graph, source, target, cutoff=5))
            
            total_effect = 0.0
            for path in paths:
                # Compute path effect (product of edge weights)
                path_effect = 1.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    path_effect *= self.causal_graph[u, v]
                
                total_effect += path_effect
            
            return total_effect
        
        except nx.NetworkXNoPath:
            return 0.0
    
    def _estimate_effect_magnitude(self, downstream_effects: Dict[int, float]) -> float:
        """Estimate overall effect magnitude."""
        if not downstream_effects:
            return 0.0
        
        # Use RMS of downstream effects as overall magnitude
        effects = list(downstream_effects.values())
        return float(np.sqrt(np.mean(np.array(effects) ** 2)))
    
    def _determine_effect_direction(self, downstream_effects: Dict[int, float]) -> str:
        """Determine overall effect direction."""
        if not downstream_effects:
            return 'none'
        
        effects = list(downstream_effects.values())
        positive_effects = sum(1 for e in effects if e > 0)
        negative_effects = sum(1 for e in effects if e < 0)
        
        if positive_effects > negative_effects * 2:
            return 'positive'
        elif negative_effects > positive_effects * 2:
            return 'negative'
        else:
            return 'mixed'
    
    def compute_confidence(self, 
                          intervention: InterventionDesign,
                          baseline_data: np.ndarray) -> float:
        """Compute confidence in intervention prediction."""
        
        # Base confidence from causal graph structure
        structure_confidence = self._compute_structure_confidence(intervention.target_variables)
        
        # Data-based confidence (if available)
        data_confidence = 1.0
        if baseline_data is not None:
            data_confidence = self._compute_data_confidence(baseline_data, intervention)
        
        # Intervention-specific confidence
        intervention_confidence = self._compute_intervention_confidence(intervention)
        
        # Combine confidences
        overall_confidence = (structure_confidence * 0.4 + 
                            data_confidence * 0.4 + 
                            intervention_confidence * 0.2)
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _compute_structure_confidence(self, target_variables: List[int]) -> float:
        """Compute confidence based on causal graph structure."""
        if self.confidence_scores is None:
            return 0.7  # Default moderate confidence
        
        # Average confidence of edges involving target variables
        confidences = []
        for target in target_variables:
            # Outgoing edges from target
            for j in range(self.causal_graph.shape[1]):
                if self.causal_graph[target, j] > 0:
                    confidences.append(self.confidence_scores[target, j])
            
            # Incoming edges to target
            for i in range(self.causal_graph.shape[0]):
                if self.causal_graph[i, target] > 0:
                    confidences.append(self.confidence_scores[i, target])
        
        return float(np.mean(confidences)) if confidences else 0.5
    
    def _compute_data_confidence(self, baseline_data: np.ndarray, intervention: InterventionDesign) -> float:
        """Compute confidence based on data quality and quantity."""
        n_samples, n_vars = baseline_data.shape
        
        # Sample size factor
        sample_confidence = min(1.0, n_samples / 1000)  # Assume 1000 samples is ideal
        
        # Data quality factor (based on variance and missing values)
        data_quality = 1.0
        for target in intervention.target_variables:
            if target < n_vars:
                var_data = baseline_data[:, target]
                if np.var(var_data) < 1e-6:  # Very low variance
                    data_quality *= 0.5
                if np.isnan(var_data).sum() > 0.1 * len(var_data):  # >10% missing
                    data_quality *= 0.7
        
        return sample_confidence * data_quality
    
    def _compute_intervention_confidence(self, intervention: InterventionDesign) -> float:
        """Compute confidence based on intervention characteristics."""
        
        # Intervention type confidence
        type_confidence = {
            'knockout': 0.9,
            'overexpression': 0.8,
            'drug_treatment': 0.7,
            'environmental': 0.6
        }.get(intervention.intervention_type, 0.5)
        
        # Strength-based confidence (moderate strengths are more reliable)
        strength = intervention.intervention_strength
        if 0.3 <= strength <= 0.8:
            strength_confidence = 1.0
        elif 0.1 <= strength < 0.3 or 0.8 < strength <= 1.0:
            strength_confidence = 0.8
        else:
            strength_confidence = 0.6
        
        return type_confidence * strength_confidence
    
    def _generate_counterfactual_outcomes(self, 
                                        intervention: InterventionDesign,
                                        baseline_data: np.ndarray) -> np.ndarray:
        """Generate counterfactual outcomes for intervention."""
        
        # Simple linear intervention model
        counterfactual_data = baseline_data.copy()
        
        for target in intervention.target_variables:
            if target < baseline_data.shape[1]:
                # Apply intervention effect
                intervention_effect = intervention.intervention_strength
                
                if intervention.intervention_type == 'knockout':
                    counterfactual_data[:, target] *= (1 - intervention_effect)
                elif intervention.intervention_type == 'overexpression':
                    counterfactual_data[:, target] *= (1 + intervention_effect)
                elif intervention.intervention_type == 'drug_treatment':
                    # Add drug effect
                    drug_effect = np.random.normal(intervention_effect, 0.1, baseline_data.shape[0])
                    counterfactual_data[:, target] += drug_effect
        
        # Propagate effects through causal network
        counterfactual_data = self._propagate_causal_effects(counterfactual_data, intervention)
        
        return counterfactual_data
    
    def _propagate_causal_effects(self, 
                                data: np.ndarray, 
                                intervention: InterventionDesign) -> np.ndarray:
        """Propagate intervention effects through causal network."""
        
        result_data = data.copy()
        
        # Get topological order for propagation
        try:
            topo_order = list(nx.topological_sort(self.nx_graph))
        except nx.NetworkXError:
            # If not a DAG, use arbitrary order
            topo_order = list(self.nx_graph.nodes())
        
        # Propagate effects in topological order
        for node in topo_order:
            if node in intervention.target_variables:
                continue  # Skip intervention targets
            
            # Compute effect from parents
            parents = list(self.nx_graph.predecessors(node))
            if parents and node < data.shape[1]:
                parent_effects = []
                for parent in parents:
                    if parent < data.shape[1]:
                        causal_strength = self.causal_graph[parent, node]
                        parent_effect = result_data[:, parent] * causal_strength
                        parent_effects.append(parent_effect)
                
                if parent_effects:
                    # Add combined parent effects
                    combined_effect = np.sum(parent_effects, axis=0)
                    result_data[:, node] += combined_effect * 0.1  # Damping factor
        
        return result_data
    
    def _compute_uncertainty_bounds(self, 
                                  downstream_effects: Dict[int, float],
                                  confidence: float) -> Tuple[float, float]:
        """Compute uncertainty bounds for effect estimates."""
        
        if not downstream_effects:
            return (0.0, 0.0)
        
        effect_magnitude = self._estimate_effect_magnitude(downstream_effects)
        
        # Uncertainty scales with (1 - confidence)
        uncertainty = effect_magnitude * (1 - confidence) * 0.5
        
        lower_bound = max(0.0, effect_magnitude - uncertainty)
        upper_bound = effect_magnitude + uncertainty
        
        return (lower_bound, upper_bound)
    
    def _generate_biological_interpretation(self, 
                                          intervention: InterventionDesign,
                                          downstream_effects: Dict[int, float]) -> str:
        """Generate biological interpretation of intervention effects."""
        
        interpretation_parts = []
        
        # Intervention description
        target_names = [self.variable_names[i] for i in intervention.target_variables 
                       if i < len(self.variable_names)]
        
        if intervention.intervention_type == 'knockout':
            interpretation_parts.append(f"Knockout of {', '.join(target_names)}")
        elif intervention.intervention_type == 'overexpression':
            interpretation_parts.append(f"Overexpression of {', '.join(target_names)}")
        elif intervention.intervention_type == 'drug_treatment':
            interpretation_parts.append(f"Drug treatment targeting {', '.join(target_names)}")
        
        # Effect description
        strong_effects = [(var, effect) for var, effect in downstream_effects.items() 
                         if abs(effect) > 0.3]
        
        if strong_effects:
            interpretation_parts.append("is predicted to strongly affect:")
            for var, effect in strong_effects[:5]:  # Top 5 effects
                var_name = self.variable_names[var] if var < len(self.variable_names) else f"Var_{var}"
                direction = "increase" if effect > 0 else "decrease"
                interpretation_parts.append(f"- {var_name} ({direction})")
        
        return " ".join(interpretation_parts)
    
    def _estimate_experimental_feasibility(self, intervention: InterventionDesign) -> float:
        """Estimate experimental feasibility of intervention."""
        
        # Base feasibility by intervention type
        base_feasibility = {
            'knockout': 0.8,  # CRISPR is well-established
            'overexpression': 0.7,  # Transfection/transduction
            'drug_treatment': 0.9,  # Usually straightforward
            'environmental': 0.6   # May require specialized conditions
        }.get(intervention.intervention_type, 0.5)
        
        # Adjust for number of targets (more targets = harder)
        target_penalty = max(0.0, 0.1 * (len(intervention.target_variables) - 1))
        
        # Adjust for intervention strength (extreme values harder)
        strength = intervention.intervention_strength
        if strength > 0.9 or strength < 0.1:
            strength_penalty = 0.2
        else:
            strength_penalty = 0.0
        
        feasibility = base_feasibility - target_penalty - strength_penalty
        return max(0.1, min(1.0, feasibility))

class DeepLearningInterventionPredictor(InterventionPredictor):
    """
    Deep learning-based intervention effect prediction.
    
    Uses neural networks trained on historical intervention data.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 input_dim: int = 100,
                 hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize deep learning predictor.
        
        Args:
            model_path: Path to pre-trained model (optional)
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build model
        self.model = self._build_model()
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.warning("No pre-trained model available - predictions may be unreliable")
    
    def _build_model(self) -> nn.Module:
        """Build neural network model for intervention prediction."""
        
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layers
        layers.extend([
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Effect magnitude
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def update_with_data(self, data: np.ndarray):
        """Update the predictor with new data by retraining the model."""
        logger.info("Retraining DeepLearningInterventionPredictor with new data.")
        # This should implement a proper retraining pipeline
        pass
    
    def predict_intervention_effect(self, 
                                  intervention: InterventionDesign,
                                  baseline_data: np.ndarray) -> InterventionResult:
        """Predict intervention effects using deep learning model."""
        
        # Encode intervention and baseline data
        input_features = self._encode_intervention(intervention, baseline_data)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
            effect_magnitude = self.model(input_tensor).item()
        
        # Compute confidence (placeholder implementation)
        confidence = self.compute_confidence(intervention, baseline_data)
        
        # Generate dummy downstream effects (would be learned from data)
        downstream_effects = self._generate_downstream_effects(
            intervention, effect_magnitude
        )
        
        return InterventionResult(
            intervention_type=intervention.intervention_type,
            target_variables=intervention.target_variables,
            effect_magnitude=effect_magnitude,
            effect_direction='positive' if effect_magnitude > 0.5 else 'negative',
            confidence=confidence,
            affected_variables=intervention.target_variables,
            downstream_effects=downstream_effects,
            biological_interpretation=f"Deep learning prediction for {intervention.intervention_id}"
        )
    
    def _encode_intervention(self, 
                           intervention: InterventionDesign,
                           baseline_data: np.ndarray) -> np.ndarray:
        """Encode intervention and baseline data for model input."""
        
        # One-hot encode intervention type
        intervention_types = ['knockout', 'overexpression', 'drug_treatment', 'environmental']
        type_encoding = np.zeros(len(intervention_types))
        if intervention.intervention_type in intervention_types:
            type_encoding[intervention_types.index(intervention.intervention_type)] = 1
        
        # Target variables encoding (binary vector)
        target_encoding = np.zeros(baseline_data.shape[1])
        for target in intervention.target_variables:
            if target < len(target_encoding):
                target_encoding[target] = 1
        
        # Intervention strength
        strength_encoding = np.array([intervention.intervention_strength])
        
        # Baseline data summary statistics
        baseline_stats = np.array([
            np.mean(baseline_data, axis=0).mean(),
            np.std(baseline_data, axis=0).mean(),
            np.median(baseline_data, axis=0).mean()
        ])
        
        # Combine all features
        features = np.concatenate([
            type_encoding,
            target_encoding,
            strength_encoding,
            baseline_stats
        ])
        
        # Pad or truncate to input_dim
        if len(features) > self.input_dim:
            features = features[:self.input_dim]
        elif len(features) < self.input_dim:
            padding = np.zeros(self.input_dim - len(features))
            features = np.concatenate([features, padding])
        
        return features
    
    def _generate_downstream_effects(self, 
                                   intervention: InterventionDesign,
                                   effect_magnitude: float) -> Dict[int, float]:
        """Generate downstream effects (placeholder implementation)."""
        
        downstream_effects = {}
        
        # Direct effects on targets
        for target in intervention.target_variables:
            downstream_effects[target] = effect_magnitude
        
        # Random downstream effects for demonstration
        # In practice, this would be learned from data
        np.random.seed(42)  # For reproducibility
        for i in range(5):  # Add 5 random downstream effects
            var_idx = np.random.randint(0, 100)
            if var_idx not in downstream_effects:
                downstream_effects[var_idx] = effect_magnitude * np.random.uniform(0.1, 0.5)
        
        return downstream_effects
    
    def compute_confidence(self, 
                          intervention: InterventionDesign,
                          baseline_data: np.ndarray) -> float:
        """Compute confidence in deep learning prediction."""
        
        # Placeholder implementation
        # In practice, this could use:
        # - Ensemble predictions
        # - Dropout-based uncertainty
        # - Calibration methods
        
        base_confidence = 0.7  # Base confidence for deep learning
        
        # Adjust based on intervention type familiarity
        type_confidence = {
            'knockout': 0.9,
            'overexpression': 0.8,
            'drug_treatment': 0.7,
            'environmental': 0.6
        }.get(intervention.intervention_type, 0.5)
        
        return base_confidence * type_confidence

class InterventionOptimizer:
    """
    Optimizer for designing optimal intervention experiments.
    
    Uses various criteria to select the most informative interventions.
    """
    
    def __init__(self, 
                 predictor: InterventionPredictor,
                 optimization_criteria: List[str] = ['information_gain', 'feasibility', 'cost']):
        """
        Initialize intervention optimizer.
        
        Args:
            predictor: Intervention effect predictor
            optimization_criteria: List of optimization criteria to use
        """
        self.predictor = predictor
        self.optimization_criteria = optimization_criteria
        self.candidate_interventions = []
        self.optimization_history = []
    
    def add_candidate_intervention(self, intervention: InterventionDesign):
        """Add a candidate intervention for optimization."""
        self.candidate_interventions.append(intervention)
    
    def optimize_intervention_selection(self, 
                                      baseline_data: np.ndarray,
                                      n_interventions: int = 10,
                                      budget_constraint: Optional[float] = None) -> List[InterventionDesign]:
        """
        Select optimal interventions using multi-criteria optimization.
        
        Args:
            baseline_data: Baseline experimental data
            n_interventions: Number of interventions to select
            budget_constraint: Maximum budget for selected interventions
            
        Returns:
            List of selected optimal interventions
        """
        
        logger.info(f"Optimizing selection of {n_interventions} interventions from {len(self.candidate_interventions)} candidates")
        
        if not self.candidate_interventions:
            logger.error("No candidate interventions available")
            return []
        
        # Score all candidate interventions
        intervention_scores = []
        
        for intervention in self.candidate_interventions:
            # Predict intervention effects
            prediction = self.predictor.predict_intervention_effect(intervention, baseline_data)
            
            # Compute multi-criteria score
            score = self._compute_intervention_score(intervention, prediction)
            
            intervention_scores.append({
                'intervention': intervention,
                'prediction': prediction,
                'score': score,
                'cost': intervention.cost_estimate or 1.0
            })
        
        # Sort by score
        intervention_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top interventions considering budget constraint
        selected_interventions = []
        total_cost = 0.0
        
        for item in intervention_scores:
            if len(selected_interventions) >= n_interventions:
                break
            
            intervention_cost = item['cost']
            
            if budget_constraint is None or total_cost + intervention_cost <= budget_constraint:
                selected_interventions.append(item['intervention'])
                total_cost += intervention_cost
        
        # Log optimization results
        logger.info(f"Selected {len(selected_interventions)} interventions with total cost: {total_cost:.2f}")
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_candidates': len(self.candidate_interventions),
            'n_selected': len(selected_interventions),
            'total_cost': total_cost,
            'budget_constraint': budget_constraint,
            'top_scores': [item['score'] for item in intervention_scores[:n_interventions]]
        })
        
        return selected_interventions
    
    def _compute_intervention_score(self, 
                                  intervention: InterventionDesign,
                                  prediction: InterventionResult) -> float:
        """Compute multi-criteria score for intervention."""
        
        scores = {}
        
        # Information gain score
        if 'information_gain' in self.optimization_criteria:
            scores['information_gain'] = self._compute_information_gain_score(prediction)
        
        # Feasibility score
        if 'feasibility' in self.optimization_criteria:
            scores['feasibility'] = prediction.experimental_feasibility or 0.5
        
        # Cost efficiency score
        if 'cost' in self.optimization_criteria:
            scores['cost'] = self._compute_cost_efficiency_score(intervention, prediction)
        
        # Novelty score
        if 'novelty' in self.optimization_criteria:
            scores['novelty'] = self._compute_novelty_score(intervention)
        
        # Combine scores with equal weights (could be made configurable)
        if scores:
            return float(np.mean(list(scores.values())))
        else:
            return 0.0
    
    def _compute_information_gain_score(self, prediction: InterventionResult) -> float:
        """Compute expected information gain from intervention."""
        
        # Information gain based on:
        # 1. Effect magnitude (larger effects are more informative)
        # 2. Number of affected variables (more comprehensive effects)
        # 3. Confidence (higher confidence = more reliable information)
        
        magnitude_score = min(1.0, prediction.effect_magnitude / 2.0)
        
        # Coverage score (how many variables are affected)
        coverage_score = min(1.0, len(prediction.affected_variables) / 20.0)
        
        # Confidence score
        confidence_score = prediction.confidence
        
        # Uncertainty reduction score (higher uncertainty means more to learn)
        uncertainty_score = 1.0 - confidence_score
        
        # Combine scores
        information_gain = (magnitude_score * 0.3 + 
                          coverage_score * 0.2 + 
                          confidence_score * 0.2 + 
                          uncertainty_score * 0.3)
        
        return information_gain
    
    def _compute_cost_efficiency_score(self, 
                                     intervention: InterventionDesign,
                                     prediction: InterventionResult) -> float:
        """Compute cost efficiency score."""
        
        cost = intervention.cost_estimate or 1.0
        expected_information = self._compute_information_gain_score(prediction)
        
        # Efficiency = information / cost
        efficiency = expected_information / max(0.1, cost)
        
        # Normalize to 0-1 range
        return min(1.0, efficiency)
    
    def _compute_novelty_score(self, intervention: InterventionDesign) -> float:
        """Compute novelty score based on previous experiments."""
        
        # Check if similar interventions have been tested before
        # This is a placeholder implementation
        
        novelty_score = 1.0  # Assume novel by default
        
        # Reduce score if similar interventions exist in history
        for hist_item in self.optimization_history:
            # This would need more sophisticated similarity computation
            # For now, just use a simple heuristic
            novelty_score *= 0.9
        
        return max(0.1, novelty_score)
    
    def generate_intervention_combinations(self, 
                                         base_interventions: List[InterventionDesign],
                                         max_combinations: int = 100) -> List[InterventionDesign]:
        """Generate combination interventions from base interventions."""
        
        logger.info(f"Generating intervention combinations from {len(base_interventions)} base interventions")
        
        combinations = []
        
        # Single interventions
        combinations.extend(base_interventions)
        
        # Pairwise combinations
        for i in range(len(base_interventions)):
            for j in range(i + 1, len(base_interventions)):
                if len(combinations) >= max_combinations:
                    break
                
                combo = self._combine_interventions(base_interventions[i], base_interventions[j])
                if combo:
                    combinations.append(combo)
        
        logger.info(f"Generated {len(combinations)} intervention combinations")
        return combinations[:max_combinations]
    
    def _combine_interventions(self, 
                             intervention1: InterventionDesign,
                             intervention2: InterventionDesign) -> Optional[InterventionDesign]:
        """Combine two interventions into a single combination intervention."""
        
        # Check compatibility
        if not self._are_interventions_compatible(intervention1, intervention2):
            return None
        
        # Create combination
        combo_id = f"{intervention1.intervention_id}_x_{intervention2.intervention_id}"
        combo_type = f"{intervention1.intervention_type}_+_{intervention2.intervention_type}"
        
        # Combine target variables
        combined_targets = list(set(intervention1.target_variables + intervention2.target_variables))
        
        # Average intervention strength (could be more sophisticated)
        combined_strength = (intervention1.intervention_strength + intervention2.intervention_strength) / 2
        
        # Estimate combined cost
        combined_cost = (intervention1.cost_estimate or 1.0) + (intervention2.cost_estimate or 1.0)
        
        return InterventionDesign(
            intervention_id=combo_id,
            intervention_type=combo_type,
            target_variables=combined_targets,
            intervention_strength=combined_strength,
            cost_estimate=combined_cost,
            duration=max(intervention1.duration or 0, intervention2.duration or 0),
            expected_outcomes={}  # Would need to be computed
        )
    
    def _are_interventions_compatible(self, 
                                    intervention1: InterventionDesign,
                                    intervention2: InterventionDesign) -> bool:
        """Check if two interventions can be combined."""
        
        # Check for conflicting targets
        targets1 = set(intervention1.target_variables)
        targets2 = set(intervention2.target_variables)
        
        # Don't combine if they target the same variables in conflicting ways
        if targets1 & targets2:
            if (intervention1.intervention_type == 'knockout' and 
                intervention2.intervention_type == 'overexpression'):
                return False
        
        # Check for incompatible types
        incompatible_pairs = [
            ('knockout', 'overexpression'),
            ('drug_treatment', 'environmental')  # Example incompatibility
        ]
        
        for type1, type2 in incompatible_pairs:
            if ((intervention1.intervention_type == type1 and intervention2.intervention_type == type2) or
                (intervention1.intervention_type == type2 and intervention2.intervention_type == type1)):
                return False
        
        return True

class ExperimentalDesignEngine:
    """
    Engine for optimal experimental design in causal intervention studies.
    
    Integrates causal discovery, intervention prediction, and optimization
    to design maximally informative experiments.
    """
    
    def __init__(self, 
                 causal_predictor: InterventionPredictor,
                 design_objectives: List[str] = ['maximize_information', 'minimize_cost']):
        """
        Initialize experimental design engine.
        
        Args:
            causal_predictor: Predictor for intervention effects
            design_objectives: List of design objectives
        """
        self.causal_predictor = causal_predictor
        self.design_objectives = design_objectives
        self.optimizer = InterventionOptimizer(causal_predictor)
        self.experiment_history = []
        self.design_constraints: Dict[str, Any] = {}
        self.prediction_history: List[Any] = []
    
    def set_design_constraints(self, constraints: Dict[str, Any]):
        """Set constraints for experimental design."""
        self.design_constraints.update(constraints)
        logger.info(f"Updated design constraints: {list(constraints.keys())}")
    
    def design_optimal_experiment_batch(self, 
                                      baseline_data: np.ndarray,
                                      candidate_interventions: List[InterventionDesign],
                                      batch_size: int = 96,
                                      budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Design an optimal batch of experiments.
        
        Args:
            baseline_data: Current experimental data
            candidate_interventions: Candidate interventions to choose from
            batch_size: Number of experiments in batch (e.g., wells in plate)
            budget: Budget constraint for batch
            
        Returns:
            Experimental design with selected interventions and controls
        """
        
        logger.info(f"Designing optimal experiment batch (size: {batch_size})")
        
        # Add candidate interventions to optimizer
        for intervention in candidate_interventions:
            self.optimizer.add_candidate_intervention(intervention)
        
        # Reserve space for controls
        n_controls = max(8, batch_size // 12)  # At least 8 controls, or ~8% of batch
        n_experiments = batch_size - n_controls
        
        logger.info(f"Allocating {n_experiments} experiments and {n_controls} controls")
        
        # Select optimal interventions
        selected_interventions = self.optimizer.optimize_intervention_selection(
            baseline_data=baseline_data,
            n_interventions=n_experiments,
            budget_constraint=budget
        )
        
        # Design control conditions
        control_conditions = self._design_control_conditions(n_controls, baseline_data)
        
        # Create experimental layout
        experimental_layout = self._create_experimental_layout(
            selected_interventions, control_conditions, batch_size
        )
        
        # Compute design metrics
        design_metrics = self._compute_design_metrics(
            selected_interventions, control_conditions, baseline_data
        )
        
        # Generate experimental protocol
        protocol = self._generate_experimental_protocol(
            selected_interventions, control_conditions
        )
        
        experiment_design = {
            'design_id': f"batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            'selected_interventions': selected_interventions,
            'control_conditions': control_conditions,
            'experimental_layout': experimental_layout,
            'design_metrics': design_metrics,
            'protocol': protocol,
            'batch_size': batch_size,
            'budget_used': sum(i.cost_estimate or 1.0 for i in selected_interventions),
            'expected_outcomes': self._predict_batch_outcomes(selected_interventions, baseline_data)
        }
        
        # Store in history
        self.experiment_history.append(experiment_design)
        
        logger.info(f"Experimental design completed: {len(selected_interventions)} interventions, {len(control_conditions)} controls")
        
        return experiment_design
    
    def _design_control_conditions(self, 
                                 n_controls: int,
                                 baseline_data: np.ndarray) -> List[Dict[str, Any]]:
        """Design appropriate control conditions."""
        
        controls = []
        
        # Negative control (no intervention)
        controls.append({
            'type': 'negative_control',
            'description': 'No intervention (vehicle only)',
            'replicates': max(3, n_controls // 3)
        })
        
        # Positive control (known strong effect)
        controls.append({
            'type': 'positive_control',
            'description': 'Known positive intervention',
            'replicates': max(2, n_controls // 4)
        })
        
        # Technical controls
        remaining_controls = n_controls - sum(c['replicates'] for c in controls)
        if remaining_controls > 0:
            controls.append({
                'type': 'technical_control',
                'description': 'Technical replicates of baseline',
                'replicates': remaining_controls
            })
        
        return controls
    
    def _create_experimental_layout(self, 
                                  interventions: List[InterventionDesign],
                                  controls: List[Dict[str, Any]],
                                  batch_size: int) -> Dict[str, Any]:
        """Create spatial layout for experiments."""
        
        # This is a simplified placeholder for a complex layout algorithm
        n_interventions = len(interventions)
        n_controls = len(controls)
        total_samples = n_interventions + n_controls
        
        # Determine plate dimensions (e.g., 96-well plate)
        n_cols = 12
        n_rows = (total_samples + n_cols - 1) // n_cols

        plate_layout: Dict[str, Dict[str, Any]] = {}
        all_samples = (
            [{'type': 'intervention', 'details': i} for i in interventions] +
            [{'type': 'control', 'details': c} for c in controls]
        )
        # Convert to numpy array for shuffling
        sample_array = np.array(all_samples, dtype=object)
        shuffle_array_inplace(sample_array)

        for i, sample in enumerate(sample_array):
            row = chr(ord('A') + (i // n_cols))
            col = i % n_cols + 1
            well_id = f"{row}{col}"
            
            if sample['type'] == 'intervention':
                intervention: InterventionDesign = sample['details']
                plate_layout[well_id] = {
                    'sample_type': 'intervention',
                    'id': str(intervention.intervention_id),
                    'intervention': intervention 
                }
            else: # control
                control: Dict[str, Any] = sample['details']
                plate_layout[well_id] = {
                    'sample_type': 'control',
                    'id': str(control['id']),
                    'control': control
                }

        # Convert intervention objects to IDs for the final layout
        plate_details = {}
        for well_id, well_info in plate_layout.items():
            plate_details[well_id] = {
                'sample_type': well_info['sample_type'],
                'id': well_info['id']
            }
        
        return {
            'plate_layout': plate_details,
            'plate_dimensions': (n_rows, n_cols),
            'total_wells': n_rows * n_cols,
            'replication_strategy': "none",
            'randomization': "full"
        }
    
    def _compute_design_metrics(self, 
                              interventions: List[InterventionDesign],
                              controls: List[Dict[str, Any]],
                              baseline_data: np.ndarray) -> Dict[str, float]:
        """Compute metrics for experimental design quality."""
        
        metrics = {}
        
        # Design diversity (how different are the interventions)
        if len(interventions) > 1:
            diversity_score = self._compute_intervention_diversity(interventions)
            metrics['diversity_score'] = diversity_score
        
        # Expected information gain
        total_information = 0.0
        for intervention in interventions:
            prediction = self.causal_predictor.predict_intervention_effect(intervention, baseline_data)
            total_information += self.optimizer._compute_information_gain_score(prediction)
        
        metrics['expected_information_gain'] = total_information / len(interventions)
        
        # Cost efficiency
        total_cost = sum(i.cost_estimate or 1.0 for i in interventions)
        metrics['cost_efficiency'] = total_information / max(1.0, total_cost)
        
        # Statistical power (placeholder)
        metrics['estimated_statistical_power'] = 0.8  # Would need proper calculation
        
        # Control ratio
        n_controls = sum(c['replicates'] for c in controls)
        metrics['control_ratio'] = n_controls / (len(interventions) + n_controls)
        
        return metrics
    
    def _compute_intervention_diversity(self, interventions: List[InterventionDesign]) -> float:
        """Compute diversity of interventions in a batch."""
        if len(interventions) < 2:
            return 1.0
            
        # Compute pairwise distances between interventions
        # (e.g., based on target variable overlap)
        n_interventions = len(interventions)
        distances = np.zeros((n_interventions, n_interventions))
        for i in range(n_interventions):
            for j in range(i + 1, n_interventions):
                set_i = set(interventions[i].target_variables)
                set_j = set(interventions[j].target_variables)
                jaccard_sim = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                distances[i, j] = 1 - jaccard_sim
        
        return float(np.mean(distances))
    
    def _generate_experimental_protocol(self, 
                                      interventions: List[InterventionDesign],
                                      controls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed experimental protocol."""
        
        protocol = {
            'title': 'Causal Intervention Experiment',
            'overview': f'Testing {len(interventions)} interventions with {sum(c["replicates"] for c in controls)} controls',
            'materials': [],
            'methods': [],
            'timeline': [],
            'data_collection': [],
            'analysis_plan': []
        }
        
        # Materials
        intervention_types = set(i.intervention_type for i in interventions)
        
        for int_type in intervention_types:
            if int_type == 'knockout':
                protocol['materials'].append('CRISPR-Cas9 system, guide RNAs')
            elif int_type == 'overexpression':
                protocol['materials'].append('Expression vectors, transfection reagents')
            elif int_type == 'drug_treatment':
                protocol['materials'].append('Compound library, vehicle controls')
            elif int_type == 'environmental':
                protocol['materials'].append('Environmental control systems')
        
        protocol['materials'].extend([
            'Cell culture media and reagents',
            'Imaging reagents (DAPI, fluorescent markers)',
            'Multi-well plates (96-well or 384-well)',
            'Automated imaging system'
        ])
        
        # Methods
        protocol['methods'] = [
            '1. Prepare cell cultures in appropriate media',
            '2. Seed cells in multi-well plates according to layout',
            '3. Allow cells to adhere (24-48 hours)',
            '4. Apply interventions according to design:',
        ]
        
        for intervention in interventions[:5]:  # Show first 5 as examples
            protocol['methods'].append(
                f'   - {intervention.intervention_id}: {intervention.intervention_type} '
                f'(strength: {intervention.intervention_strength:.2f})'
            )
        
        protocol['methods'].extend([
            '5. Incubate for specified duration',
            '6. Fix and stain cells for imaging',
            '7. Acquire images using automated microscopy',
            '8. Extract quantitative features from images'
        ])
        
        # Timeline
        protocol['timeline'] = [
            'Day 0: Cell seeding and setup',
            'Day 1-2: Cell adherence and stabilization',
            'Day 2: Apply interventions',
            'Day 3-5: Incubation period (varies by intervention)',
            'Day 5: Fixation and staining',
            'Day 6: Image acquisition',
            'Day 7-10: Data analysis and interpretation'
        ]
        
        # Data collection
        protocol['data_collection'] = [
            'High-content imaging (multiple channels)',
            'Morphological feature extraction',
            'Quantitative phenotypic measurements',
            'Quality control metrics',
            'Metadata recording (plate, well, conditions)'
        ]
        
        # Analysis plan
        protocol['analysis_plan'] = [
            'Image preprocessing and quality control',
            'Feature extraction and normalization',
            'Statistical analysis of intervention effects',
            'Causal inference and pathway analysis',
            'Visualization and interpretation'
        ]
        
        return protocol
    
    def _predict_batch_outcomes(self, 
                              interventions: List[InterventionDesign],
                              baseline_data: np.ndarray) -> Dict[str, Any]:
        """Predict outcomes for the entire batch of experiments."""
        
        predicted_outcomes = {}
        
        for intervention in interventions:
            prediction = self.causal_predictor.predict_intervention_effect(
                intervention, baseline_data
            )
            
            # Combine scores
            feasibility = prediction.experimental_feasibility or 0.0
            weighted_score = (
                (prediction.confidence * prediction.effect_magnitude) + 
                (feasibility * 0.5)
            )

            predicted_outcomes[intervention.intervention_id] = {
                'predicted_effect': prediction.effect_magnitude,
                'confidence': prediction.confidence,
                'affected_variables': prediction.affected_variables,
                'success_probability': prediction.confidence * feasibility
            }
        
        # Aggregate statistics
        effect_magnitudes = [p['predicted_effect'] for p in predicted_outcomes.values()]
        confidences = [p['confidence'] for p in predicted_outcomes.values()]
        
        predicted_outcomes['_summary'] = {
            'total_interventions': len(interventions),
            'mean_effect_magnitude': np.mean(effect_magnitudes),
            'mean_confidence': np.mean(confidences),
            'high_impact_interventions': sum(1 for e in effect_magnitudes if e > 0.5),
            'high_confidence_interventions': sum(1 for c in confidences if c > 0.8)
        }
        
        return predicted_outcomes
    
    def update_with_experimental_results(self, 
                                       experiment_results: Dict[str, Any]):
        """Update causal models with new experimental data."""
        
        logger.info("Updating models with new experimental results...")
        
        # Update the causal predictor
        if 'full_dataset' in experiment_results:
            self.causal_predictor.update_with_data(experiment_results['full_dataset'])
        
        # Analyze prediction accuracy
        self._analyze_prediction_accuracy(experiment_results)
        
        self.prediction_history.append(experiment_results)
    
    def _analyze_prediction_accuracy(self, experiment_results: Dict[str, Any]):
        """Analyze how well predictions matched actual results."""
        
        if 'predictions' not in experiment_results or 'actual_results' not in experiment_results:
            logger.warning("Cannot analyze prediction accuracy - missing data")
            return
        
        predictions = experiment_results['predictions']
        actual_results = experiment_results['actual_results']
        
        accuracies = []
        
        for intervention_id in predictions:
            if intervention_id in actual_results:
                pred_effect = predictions[intervention_id]['predicted_effect']
                actual_effect = actual_results[intervention_id]['effect_magnitude']
                
                # Compute relative error
                if actual_effect != 0:
                    relative_error = abs(pred_effect - actual_effect) / abs(actual_effect)
                    accuracy = max(0, 1 - relative_error)
                    accuracies.append(accuracy)
        
        if accuracies:
            mean_accuracy = np.mean(accuracies)
            logger.info(f"Prediction accuracy: {mean_accuracy:.3f}")
            
            # Store accuracy metrics
            if 'prediction_accuracy_history' not in self.design_constraints:
                self.design_constraints['prediction_accuracy_history'] = []
            
            self.design_constraints['prediction_accuracy_history'].append({
                'timestamp': pd.Timestamp.now(),
                'mean_accuracy': mean_accuracy,
                'n_predictions': len(accuracies)
            })

# Utility functions for intervention analysis

def create_standard_intervention_library(n_variables: int,
                                        intervention_types: Optional[List[str]] = None) -> List[InterventionDesign]:
    """
    Create a standard library of candidate interventions.
    
    Args:
        n_variables: Number of variables in the system
        intervention_types: Types of interventions to create
        
    Returns:
        List of intervention designs
    """
    
    if intervention_types is None:
        intervention_types = ['knockout', 'overexpression']

    interventions = []
    for var_idx in range(n_variables):
        for int_type in intervention_types:
            for strength in [0.3, 0.5, 0.8]:  # Different intervention strengths
                intervention = InterventionDesign(
                    intervention_id=f"{int_type}_{var_idx}_s{strength}",
                    intervention_type=int_type,
                    target_variables=[var_idx],
                    intervention_strength=strength,
                    cost_estimate=np.random.uniform(50, 200),
                    duration=24.0,  # 24 hours
                    expected_outcomes={}
                )
                interventions.append(intervention)
    
    logger.info(f"Created standard intervention library with {len(interventions)} interventions")
    return interventions

def simulate_intervention_experiment(intervention: InterventionDesign,
                                   baseline_data: np.ndarray,
                                   noise_level: float = 0.1) -> Dict[str, Any]:
    """Simulate the results of an intervention experiment."""
    
    n_samples, n_vars = baseline_data.shape
    
    # Create perturbed data
    perturbed_data = baseline_data.copy()
    
    # Apply intervention effects
    for target in intervention.target_variables:
        if target < n_vars:
            if intervention.intervention_type == 'knockout':
                # Reduce expression
                perturbed_data[:, target] *= (1 - intervention.intervention_strength)
            elif intervention.intervention_type == 'overexpression':
                # Increase expression
                perturbed_data[:, target] *= (1 + intervention.intervention_strength)
            elif intervention.intervention_type == 'drug_treatment':
                # Add drug effect with some variability
                drug_effect = np.random.normal(
                    intervention.intervention_strength, 
                    intervention.intervention_strength * 0.2, 
                    n_samples
                )
                perturbed_data[:, target] += drug_effect
    
    # Add downstream effects (simplified)
    for target in intervention.target_variables:
        if target < n_vars - 1:
            # Affect next variable as downstream effect
            downstream_effect = intervention.intervention_strength * 0.3
            perturbed_data[:, target + 1] += downstream_effect
    
    # Add experimental noise
    noise = np.random.normal(0, noise_level, perturbed_data.shape)
    perturbed_data += noise
    
    # Compute effect statistics
    effect_magnitude = np.mean(np.abs(perturbed_data - baseline_data))
    
    # Simulate measurement confidence
    confidence = np.random.uniform(0.7, 0.95)
    
    return {
        'intervention_id': intervention.intervention_id,
        'perturbed_data': perturbed_data,
        'effect_magnitude': effect_magnitude,
        'confidence': confidence,
        'success': True,
        'metadata': {
            'simulation_noise': noise_level,
            'n_samples': n_samples,
            'n_variables': n_vars
        }
    }

def validate_intervention_predictions(predictor: InterventionPredictor,
                                    test_interventions: List[InterventionDesign],
                                    test_data: np.ndarray,
                                    simulate_results: bool = True) -> Dict[str, Any]:
    """Validate intervention predictions against simulated or real results."""
    
    logger.info(f"Validating intervention predictor on {len(test_interventions)} interventions")
    
    validation_results = {
        'predictions': {},
        'actual_results': {},
        'accuracy_metrics': {},
        'validation_summary': {}
    }
    
    # Make predictions
    for intervention in test_interventions:
        prediction = predictor.predict_intervention_effect(intervention, test_data)
        validation_results['predictions'][intervention.intervention_id] = {
            'effect_magnitude': prediction.effect_magnitude,
            'confidence': prediction.confidence,
            'affected_variables': prediction.affected_variables
        }
        
        # Get actual results (simulated or experimental)
        if simulate_results:
            actual_result = simulate_intervention_experiment(intervention, test_data)
        else:
            # Would load actual experimental results
            actual_result = {'effect_magnitude': 0.0, 'confidence': 0.0}  # Placeholder
        
        validation_results['actual_results'][intervention.intervention_id] = actual_result
    
    # Compute accuracy metrics
    effect_errors = []
    confidence_calibration = []
    
    for intervention_id in validation_results['predictions']:
        pred = validation_results['predictions'][intervention_id]
        actual = validation_results['actual_results'][intervention_id]
        
        # Effect magnitude error
        pred_effect = pred['effect_magnitude']
        actual_effect = actual['effect_magnitude']
        
        if actual_effect != 0:
            relative_error = abs(pred_effect - actual_effect) / actual_effect
            effect_errors.append(relative_error)
        
        # Confidence calibration
        pred_confidence = pred['confidence']
        actual_confidence = actual.get('confidence', 0.5)
        confidence_error = abs(pred_confidence - actual_confidence)
        confidence_calibration.append(confidence_error)
    
    # Summary statistics
    validation_results['accuracy_metrics'] = {
        'mean_effect_error': np.mean(effect_errors) if effect_errors else 0.0,
        'std_effect_error': np.std(effect_errors) if effect_errors else 0.0,
        'mean_confidence_error': np.mean(confidence_calibration),
        'r2_score': compute_r2_score(validation_results) if len(effect_errors) > 1 else 0.0
    }
    
    validation_results['validation_summary'] = {
        'n_interventions_tested': len(test_interventions),
        'prediction_accuracy': 1 - np.mean(effect_errors) if effect_errors else 0.0,
        'confidence_calibration': 1 - np.mean(confidence_calibration),
        'overall_performance': (validation_results['accuracy_metrics']['r2_score'] + 
                              (1 - np.mean(effect_errors) if effect_errors else 0.0)) / 2
    }
    
    logger.info(f"Validation completed - Overall performance: {validation_results['validation_summary']['overall_performance']:.3f}")
    
    return validation_results

def compute_r2_score(validation_results: Dict[str, Any]) -> float:
    """Compute R² score for effect magnitude predictions."""
    
    predictions = validation_results['predictions']
    actual_results = validation_results['actual_results']
    
    pred_effects = []
    actual_effects = []
    
    for intervention_id in predictions:
        if intervention_id in actual_results:
            pred_effects.append(predictions[intervention_id]['effect_magnitude'])
            actual_effects.append(actual_results[intervention_id]['effect_magnitude'])
    
    if len(pred_effects) < 2:
        return 0.0
    
    # Compute R²
    pred_effects = np.array(pred_effects)
    actual_effects = np.array(actual_effects)
    
    if len(actual_effects) == 0:
        return 0.0
    
    ss_res = np.sum((actual_effects - pred_effects) ** 2)
    ss_tot = np.sum((actual_effects - np.mean(actual_effects)) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return float(r_squared)

def shuffle_array_inplace(arr: Union[NDArray, List]) -> None:
    """Safely shuffle array in place."""
    if isinstance(arr, np.ndarray):
        # Use numpy's random generator for proper in-place shuffling
        rng = np.random.default_rng()
        rng.shuffle(arr)
    else:
        # Use standard random for lists
        import random
        random.shuffle(arr)

# --------------------
# Fallback metric utilities (appended by OpenPerturbation CI)
# --------------------

# Ensure fallback definitions exist in environments without scikit-learn.
try:
    _sk_accuracy  # type: ignore
except NameError:  # pragma: no cover
    def _sk_accuracy(y_true, y_pred, **kwargs):  # type: ignore
        """Simple accuracy when scikit-learn is unavailable."""
        return float((np.asarray(y_true) == np.asarray(y_pred)).mean())

    def _sk_mse(y_true, y_pred, **kwargs):  # type: ignore
        """Simple MSE when scikit-learn is unavailable."""
        diff = np.asarray(y_true) - np.asarray(y_pred)
        return float(np.mean(diff ** 2))

# Public aliases used throughout the codebase
accuracy_score = _sk_accuracy  # type: ignore
mean_squared_error = _sk_mse  # type: ignore