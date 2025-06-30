"""
Active learning for experimental design in perturbation biology.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Machine learning imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    RandomForestClassifier = None
    GaussianProcessRegressor = None

logger = logging.getLogger(__name__)


@dataclass 
class ExperimentCandidate:
    """Candidate experiment for active learning selection."""
    experiment_id: str
    conditions: Dict[str, Any]
    predicted_outcome: Optional[float] = None
    uncertainty: Optional[float] = None
    acquisition_score: Optional[float] = None
    cost: float = 1.0
    feasibility: float = 1.0


@dataclass
class ActiveLearningResult:
    """Result from active learning experiment selection."""
    selected_experiments: List[ExperimentCandidate]
    acquisition_scores: List[float]
    total_cost: float
    expected_information_gain: float
    selection_strategy: str


class AcquisitionFunction(ABC):
    """Base class for acquisition functions in active learning."""
    
    @abstractmethod
    def compute_score(self, 
                     candidates: List[ExperimentCandidate],
                     model: Any,
                     observed_data: np.ndarray) -> List[float]:
        """Compute acquisition scores for candidates."""
        pass


class UncertaintyAcquisition(AcquisitionFunction):
    """Acquisition function based on prediction uncertainty."""
    
    def compute_score(self, 
                     candidates: List[ExperimentCandidate],
                     model: Any,
                     observed_data: np.ndarray) -> List[float]:
        """Compute uncertainty-based acquisition scores."""
        
        scores = []
        for candidate in candidates:
            if candidate.uncertainty is not None:
                # Higher uncertainty = higher score
                score = float(candidate.uncertainty)
            else:
                # Default uncertainty if not provided
                score = 0.5
            
            scores.append(score)
        
        return scores


class ExpectedImprovementAcquisition(AcquisitionFunction):
    """Acquisition function based on expected improvement."""
    
    def __init__(self, xi: float = 0.01):
        """Initialize with exploration parameter."""
        self.xi = xi
    
    def compute_score(self, 
                     candidates: List[ExperimentCandidate],
                     model: Any,
                     observed_data: np.ndarray) -> List[float]:
        """Compute expected improvement scores."""
        
        if observed_data.size == 0:
            # No observed data, use uncertainty
            return [0.5 for _ in candidates]
        
        # Find current best
        current_best = np.max(observed_data[:, -1]) if observed_data.shape[1] > 0 else 0.0
        
        scores = []
        for candidate in candidates:
            if candidate.predicted_outcome is not None and candidate.uncertainty is not None:
                mean = candidate.predicted_outcome
                std = candidate.uncertainty
                
                if std > 0:
                    z = (mean - current_best - self.xi) / std
                    ei = (mean - current_best - self.xi) * self._normal_cdf(z) + std * self._normal_pdf(z)
                    scores.append(max(0.0, float(ei)))
                else:
                    scores.append(0.0)
            else:
                scores.append(0.1)  # Small default score
        
        return scores
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))
    
    def _normal_pdf(self, x: float) -> float:
        """Normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class DiversityAcquisition(AcquisitionFunction):
    """Acquisition function promoting diverse experiment selection."""
    
    def compute_score(self, 
                     candidates: List[ExperimentCandidate],
                     model: Any,
                     observed_data: np.ndarray) -> List[float]:
        """Compute diversity-based acquisition scores."""
        
        scores = []
        
        for i, candidate in enumerate(candidates):
            # Compute distance to other candidates
            min_distance = float('inf')
            
            for j, other_candidate in enumerate(candidates):
                if i != j:
                    distance = self._compute_candidate_distance(candidate, other_candidate)
                    min_distance = min(min_distance, distance)
            
            # Higher distance = higher diversity score
            if min_distance == float('inf'):
                scores.append(1.0)
            else:
                scores.append(float(min_distance))
        
        return scores
    
    def _compute_candidate_distance(self, 
                                  candidate1: ExperimentCandidate,
                                  candidate2: ExperimentCandidate) -> float:
        """Compute distance between two experiment candidates."""
        
        # Simple distance based on conditions
        distance = 0.0
        all_keys = set(candidate1.conditions.keys()) | set(candidate2.conditions.keys())
        
        for key in all_keys:
            val1 = candidate1.conditions.get(key, 0)
            val2 = candidate2.conditions.get(key, 0)
            
            # Convert to numeric if possible
            try:
                val1_num = float(val1)
                val2_num = float(val2)
                distance += (val1_num - val2_num) ** 2
            except (ValueError, TypeError):
                # Categorical variables
                distance += 0 if val1 == val2 else 1
        
        return np.sqrt(distance)


class ActiveLearningEngine:
    """Engine for active learning in experimental design."""
    
    def __init__(self,
                 acquisition_function: Optional[AcquisitionFunction] = None,
                 model_type: str = "gaussian_process",
                 random_state: int = 42):
        """Initialize active learning engine."""
        
        self.acquisition_function = acquisition_function or UncertaintyAcquisition()
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.observed_experiments: List[ExperimentCandidate] = []
        self.observed_outcomes: List[float] = []
        
    def fit_model(self, 
                  experiments: List[ExperimentCandidate],
                  outcomes: List[float]) -> None:
        """Fit predictive model on observed experiments."""
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using dummy model")
            self.model = DummyModel()
            return
        
        # Convert experiments to feature matrix
        X = self._experiments_to_features(experiments)
        y = np.array(outcomes)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No training data available")
            self.model = DummyModel()
            return
        
        # Initialize model
        if self.model_type == "gaussian_process" and GaussianProcessRegressor is not None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                random_state=self.random_state
            )
        elif self.model_type == "random_forest" and RandomForestRegressor is not None:
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            logger.warning(f"Model type {self.model_type} not available, using dummy model")
            self.model = DummyModel()
            return
        
        try:
            self.model.fit(X, y)
            logger.info(f"Fitted {self.model_type} model on {len(experiments)} experiments")
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            self.model = DummyModel()
    
    def predict_outcomes(self, 
                        candidates: List[ExperimentCandidate]) -> Tuple[List[float], List[float]]:
        """Predict outcomes and uncertainties for candidate experiments."""
        
        if self.model is None:
            logger.warning("No model fitted, returning default predictions")
            n = len(candidates)
            return [0.5] * n, [0.5] * n
        
        X = self._experiments_to_features(candidates)
        
        if len(X) == 0:
            n = len(candidates)
            return [0.5] * n, [0.5] * n
        
        try:
            if hasattr(self.model, 'predict') and hasattr(self.model, 'predict_std'):
                # Gaussian Process
                predictions, std = self.model.predict(X, return_std=True)
                return predictions.tolist(), std.tolist()
            elif hasattr(self.model, 'predict'):
                # Other models
                predictions = self.model.predict(X)
                # Estimate uncertainty from prediction variance
                uncertainties = [0.1] * len(predictions)  # Default uncertainty
                return predictions.tolist(), uncertainties
            else:
                # Dummy model
                n = len(candidates)
                return [0.5] * n, [0.5] * n
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            n = len(candidates)
            return [0.5] * n, [0.5] * n
    
    def select_experiments(self,
                          candidates: List[ExperimentCandidate],
                          n_experiments: int = 10,
                          budget_constraint: Optional[float] = None) -> ActiveLearningResult:
        """Select experiments using active learning."""
        
        # Predict outcomes and uncertainties
        predictions, uncertainties = self.predict_outcomes(candidates)
        
        # Update candidates with predictions
        for i, candidate in enumerate(candidates):
            if i < len(predictions):
                candidate.predicted_outcome = predictions[i]
            if i < len(uncertainties):
                candidate.uncertainty = uncertainties[i]
        
        # Compute acquisition scores
        observed_data = self._get_observed_data()
        acquisition_scores = self.acquisition_function.compute_score(
            candidates, self.model, observed_data
        )
        
        # Update candidates with scores
        for candidate, score in zip(candidates, acquisition_scores):
            candidate.acquisition_score = score
        
        # Select experiments
        selected_experiments = self._select_top_experiments(
            candidates, n_experiments, budget_constraint
        )
        
        # Compute metrics
        total_cost = sum(exp.cost for exp in selected_experiments)
        expected_info_gain = sum(exp.acquisition_score or 0 for exp in selected_experiments)
        
        return ActiveLearningResult(
            selected_experiments=selected_experiments,
            acquisition_scores=[exp.acquisition_score or 0 for exp in selected_experiments],
            total_cost=total_cost,
            expected_information_gain=expected_info_gain,
            selection_strategy=type(self.acquisition_function).__name__
        )
    
    def update_with_results(self,
                           experiments: List[ExperimentCandidate],
                           outcomes: List[float]) -> None:
        """Update model with new experimental results."""
        
        # Add to observed data
        self.observed_experiments.extend(experiments)
        self.observed_outcomes.extend(outcomes)
        
        # Refit model
        self.fit_model(self.observed_experiments, self.observed_outcomes)
    
    def _experiments_to_features(self, 
                               experiments: List[ExperimentCandidate]) -> np.ndarray:
        """Convert experiments to feature matrix."""
        
        if not experiments:
            return np.array([])
        
        # Get all unique condition keys
        all_keys = set()
        for exp in experiments:
            all_keys.update(exp.conditions.keys())
        
        all_keys = sorted(list(all_keys))
        
        # Convert to feature matrix
        features = []
        for exp in experiments:
            feature_vector = []
            for key in all_keys:
                value = exp.conditions.get(key, 0)
                try:
                    feature_vector.append(float(value))
                except (ValueError, TypeError):
                    # Categorical - simple hash encoding
                    feature_vector.append(float(hash(str(value)) % 1000))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _get_observed_data(self) -> np.ndarray:
        """Get observed experimental data."""
        
        if not self.observed_experiments or not self.observed_outcomes:
            return np.array([])
        
        X = self._experiments_to_features(self.observed_experiments)
        y = np.array(self.observed_outcomes).reshape(-1, 1)
        
        if X.size == 0:
            return np.array([])
        
        return np.hstack([X, y])
    
    def _select_top_experiments(self,
                              candidates: List[ExperimentCandidate],
                              n_experiments: int,
                              budget_constraint: Optional[float]) -> List[ExperimentCandidate]:
        """Select top experiments considering budget and feasibility."""
        
        # Sort by acquisition score (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (x.acquisition_score or 0) * x.feasibility,
            reverse=True
        )
        
        selected = []
        total_cost = 0.0
        
        for candidate in sorted_candidates:
            if len(selected) >= n_experiments:
                break
            
            if budget_constraint is not None:
                if total_cost + candidate.cost > budget_constraint:
                    continue
            
            selected.append(candidate)
            total_cost += candidate.cost
        
        return selected


class DummyModel:
    """Dummy model for when scikit-learn is not available."""
    
    def fit(self, X, y):
        """Dummy fit method."""
        pass
    
    def predict(self, X):
        """Dummy predict method."""
        return np.random.random(len(X)) if len(X) > 0 else np.array([])


class BatchExperimentDesigner:
    """Designer for batch experiments using active learning."""
    
    def __init__(self,
                 active_learner: ActiveLearningEngine,
                 batch_size: int = 96):
        """Initialize batch experiment designer."""
        
        self.active_learner = active_learner
        self.batch_size = batch_size
        self.design_history: List[Dict[str, Any]] = []
    
    def design_experiment_batch(self,
                              candidate_pool: List[ExperimentCandidate],
                              objectives: List[str] = None,
                              constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Design a batch of experiments."""
        
        objectives = objectives or ["maximize_information"]
        constraints = constraints or {}
        
        # Select experiments using active learning
        result = self.active_learner.select_experiments(
            candidate_pool,
            n_experiments=self.batch_size,
            budget_constraint=constraints.get("budget")
        )
        
        # Create batch layout
        batch_layout = self._create_batch_layout(result.selected_experiments)
        
        # Generate experimental protocol
        protocol = self._generate_protocol(result.selected_experiments, constraints)
        
        design_result = {
            'batch_id': f"batch_{len(self.design_history) + 1}",
            'selected_experiments': result.selected_experiments,
            'batch_layout': batch_layout,
            'protocol': protocol,
            'expected_outcomes': [exp.predicted_outcome for exp in result.selected_experiments],
            'acquisition_scores': result.acquisition_scores,
            'total_cost': result.total_cost,
            'information_gain': result.expected_information_gain,
            'design_objectives': objectives,
            'constraints': constraints
        }
        
        self.design_history.append(design_result)
        return design_result
    
    def _create_batch_layout(self, 
                           experiments: List[ExperimentCandidate]) -> Dict[str, Any]:
        """Create physical batch layout for experiments."""
        
        # Simple grid layout
        n_experiments = len(experiments)
        n_rows = int(np.ceil(np.sqrt(n_experiments)))
        n_cols = int(np.ceil(n_experiments / n_rows))
        
        layout = {
            'format': 'grid',
            'dimensions': {'rows': n_rows, 'cols': n_cols},
            'assignments': {},
            'controls': []
        }
        
        # Assign experiments to positions
        for i, exp in enumerate(experiments):
            row = i // n_cols
            col = i % n_cols
            position = f"{chr(65 + row)}{col + 1}"  # A1, A2, B1, etc.
            
            layout['assignments'][position] = {
                'experiment_id': exp.experiment_id,
                'conditions': exp.conditions,
                'type': 'experimental'
            }
        
        # Add control positions
        n_controls = max(1, n_experiments // 10)  # 10% controls
        for i in range(n_controls):
            row = (n_experiments + i) // n_cols
            col = (n_experiments + i) % n_cols
            position = f"{chr(65 + row)}{col + 1}"
            
            layout['assignments'][position] = {
                'experiment_id': f"control_{i+1}",
                'conditions': {'treatment': 'vehicle_control'},
                'type': 'control'
            }
            layout['controls'].append(position)
        
        return layout
    
    def _generate_protocol(self,
                          experiments: List[ExperimentCandidate],
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experimental protocol."""
        
        protocol = {
            'title': f'Active Learning Experiment Batch - {len(experiments)} conditions',
            'objective': 'Optimize experimental design using active learning',
            'timeline': {
                'preparation': '1-2 days',
                'execution': '3-7 days',
                'analysis': '1-3 days'
            },
            'materials': [
                'Cell culture medium',
                'Treatment compounds',
                'Analysis reagents'
            ],
            'procedure': [
                '1. Prepare experimental conditions according to design',
                '2. Seed cells in designated well positions',
                '3. Apply treatments as specified in batch layout',
                '4. Incubate for specified time periods',
                '5. Collect samples for analysis',
                '6. Perform measurements and data collection'
            ],
            'quality_controls': [
                'Include vehicle controls',
                'Randomize treatment application',
                'Use technical replicates where possible'
            ],
            'data_collection': {
                'measurements': ['viability', 'expression', 'phenotype'],
                'timepoints': constraints.get('timepoints', ['24h', '48h']),
                'replicates': constraints.get('replicates', 3)
            }
        }
        
        return protocol


def create_experiment_candidates(
    parameter_space: Dict[str, List[Any]],
    n_candidates: int = 1000,
    random_state: int = 42
) -> List[ExperimentCandidate]:
    """Create experiment candidates from parameter space."""
    
    np.random.seed(random_state)
    candidates = []
    
    param_names = list(parameter_space.keys())
    
    for i in range(n_candidates):
        conditions = {}
        for param_name in param_names:
            param_values = parameter_space[param_name]
            conditions[param_name] = np.random.choice(param_values)
        
        candidate = ExperimentCandidate(
            experiment_id=f"exp_{i+1:04d}",
            conditions=conditions,
            cost=np.random.uniform(0.5, 2.0),  # Random cost
            feasibility=np.random.uniform(0.7, 1.0)  # Random feasibility
        )
        
        candidates.append(candidate)
    
    return candidates


def run_active_learning_experiment(
    parameter_space: Dict[str, List[Any]],
    n_initial_experiments: int = 10,
    n_iterations: int = 5,
    batch_size: int = 10,
    acquisition_function: str = "uncertainty"
) -> Dict[str, Any]:
    """Run complete active learning experiment."""
    
    # Create candidate pool
    candidates = create_experiment_candidates(parameter_space, n_candidates=1000)
    
    # Initialize acquisition function
    if acquisition_function == "uncertainty":
        acq_func = UncertaintyAcquisition()
    elif acquisition_function == "expected_improvement":
        acq_func = ExpectedImprovementAcquisition()
    elif acquisition_function == "diversity":
        acq_func = DiversityAcquisition()
    else:
        acq_func = UncertaintyAcquisition()
    
    # Initialize active learning engine
    engine = ActiveLearningEngine(
        acquisition_function=acq_func,
        model_type="gaussian_process"
    )
    
    # Initial random experiments
    initial_experiments = np.random.choice(candidates, n_initial_experiments, replace=False).tolist()
    initial_outcomes = [np.random.random() for _ in initial_experiments]  # Simulate outcomes
    
    engine.update_with_results(initial_experiments, initial_outcomes)
    
    # Active learning iterations
    results = {
        'iterations': [],
        'total_experiments': n_initial_experiments,
        'final_model_performance': None
    }
    
    for iteration in range(n_iterations):
        # Select next batch
        remaining_candidates = [c for c in candidates if c not in engine.observed_experiments]
        
        if len(remaining_candidates) < batch_size:
            break
        
        selection_result = engine.select_experiments(
            remaining_candidates,
            n_experiments=batch_size
        )
        
        # Simulate experimental outcomes
        simulated_outcomes = []
        for exp in selection_result.selected_experiments:
            # Simple simulation based on conditions
            outcome = sum(hash(str(v)) % 100 for v in exp.conditions.values()) / 100.0
            outcome += np.random.normal(0, 0.1)  # Add noise
            simulated_outcomes.append(outcome)
        
        # Update model
        engine.update_with_results(selection_result.selected_experiments, simulated_outcomes)
        
        # Record iteration results
        iteration_result = {
            'iteration': iteration + 1,
            'selected_experiments': len(selection_result.selected_experiments),
            'total_cost': selection_result.total_cost,
            'information_gain': selection_result.expected_information_gain,
            'best_outcome': max(simulated_outcomes) if simulated_outcomes else 0
        }
        
        results['iterations'].append(iteration_result)
        results['total_experiments'] += len(selection_result.selected_experiments)
    
    return results
