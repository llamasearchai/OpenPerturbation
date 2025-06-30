"""
causal_discovery_engine.py
================================================
Causal-discovery algorithms for perturbation biology.

This module unifies a range of constraint-based, score-based, optimisation-based
and deep-learning causal structure-learning methods behind a single
CausalDiscoveryEngine class, plus a lightweight CausalInferenceEngine for
post-hoc intervention simulation.

Optional extras (install only what you need):
    pip install causal-learn pgmpy torch networkx seaborn matplotlib scikit-learn
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Optional heavy/ML deps ----------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn: # type: ignore
        Module = object
        ModuleList = list
        Parameter = object
        def MSELoss(self): pass

# causal-learn --------------------------------------------------------------- #
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.FCMBased import DirectLiNGAM
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    pc = ges = DirectLiNGAM = None  # type: ignore # graceful fall-back

# pgmpy ---------------------------------------------------------------------- #
try:
    from pgmpy.estimators import HillClimbSearch
    from pgmpy.models import BayesianNetwork
    try:  # pgmpy has renamed BicScore a few times
        from pgmpy.estimators import BicScore
    except ImportError:                                        # ↳ old alias
        from pgmpy.estimators import BICScore as BicScore      # type: ignore
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    HillClimbSearch = BayesianNetwork = BicScore = None  # type: ignore

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
if not logger.handlers:   # avoid duplicate handlers in notebooks
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _graph_to_adjacency(graph_like: Any, n_vars: Optional[int] = None) -> np.ndarray:
    """
    Convert a Graph-like object from causal-learn / NetworkX / pgmpy
    into a dense NumPy adjacency matrix.
    """
    if graph_like is None:
        return np.zeros((0, 0))

    # 1) causal-learn Graph
    if hasattr(graph_like, "graph"):  # causal-learn graphs expose .graph (numpy)
        adj = np.asarray(graph_like.graph, dtype=float)
        return adj

    # 2) NetworkX-style object (pgmpy returns one too)
    if hasattr(graph_like, "edges"):
        if n_vars is None:
            # Try to infer #nodes from max node id
            nodes = list(graph_like.nodes())
            n_vars = int(max(nodes)) + 1 if nodes else 0
        adj = np.zeros((n_vars, n_vars), dtype=float)
        for u, v in graph_like.edges():
            # Cast node labels to int if they are strings like "Var_3"
            if isinstance(u, str) and u.startswith("Var_"):
                u = int(u.split("_")[1])
            if isinstance(v, str) and v.startswith("Var_"):
                v = int(v.split("_")[1])
            adj[int(u), int(v)] = 1.0
        return adj

    raise ValueError("Unrecognised graph_like object")

# --------------------------------------------------------------------------- #
# Main discovery engine
# --------------------------------------------------------------------------- #
class CausalDiscoveryEngine:
    """
    High-level façade for running multiple causal-structure-learning algorithms.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.method: str = config.get("discovery_method", "pc").lower()
        self.significance_level: float = config.get("significance_level", 0.05)
        self.max_conditioning_set_size: int = config.get("max_conditioning_set_size", 3)
        self.bootstrap_samples: int = config.get("bootstrap_samples", 100)

        self.discovered_graph: Optional[np.ndarray] = None
        self.confidence_scores: Optional[np.ndarray] = None

    # --------------------------------------------------------------------- #
    # Public entry-point
    # --------------------------------------------------------------------- #
    def discover_causal_structure(
        self,
        data: Union[np.ndarray, "torch.Tensor", pd.DataFrame],
        interventions: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the configured discovery algorithm and return a rich result dict.
        """
        logger.info("SEARCH:  Starting causal discovery with method: %s", self.method)

        X, var_names = self._preprocess_data(data, variable_names)

        # Dispatch
        dispatch = {
            "pc": self._discover_with_pc,
            "ges": self._discover_with_ges,
            "lingam": self._discover_with_lingam,
            "notears": self._discover_with_notears,
            "bayesian_network": self._discover_with_bayesian_network,
            "deep_causal": self._discover_with_deep_learning,
            "correlation": self._discover_with_correlation,
        }
        result = dispatch.get(self.method, self._discover_with_correlation)(X, interventions)

        # House-keeping
        result.update({
            "method": self.method,
            "variable_names": var_names,
            "n_samples": X.shape[0],
            "n_variables": X.shape[1],
            "config": self.config,
        })
        self.discovered_graph = result["adjacency_matrix"]
        self.confidence_scores = result.get("confidence_scores")

        # Post-hoc network analysis
        result["analysis"] = self._analyze_causal_structure(result)
        logger.info("SUCCESS:  Discovery done: %d edges",
                    result["analysis"]["num_edges"])
        return result

    # --------------------------------------------------------------------- #
    # Pre-processing
    # --------------------------------------------------------------------- #
    def _preprocess_data(
        self,
        data: Union[np.ndarray, "torch.Tensor", pd.DataFrame],
        variable_names: Optional[List[str]]
    ) -> Tuple[np.ndarray, List[str]]:

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        elif isinstance(data, pd.DataFrame):
            data_np = data.values
            variable_names = variable_names or list(data.columns)
        else:
            data_np = np.asarray(data)

        if data_np.ndim != 2:
            raise ValueError(f"Data must be 2D, but got shape {data_np.shape}")

        if variable_names is None:
            variable_names = [f"Var_{i}" for i in range(data_np.shape[1])]

        # Missing values → mean impute
        if np.isnan(data_np).any():
            from sklearn.impute import SimpleImputer
            data_np = SimpleImputer(strategy="mean").fit_transform(data_np)
            logger.warning("Missing values detected – applied mean imputation.")

        # Standardise (optional)
        if self.config.get("standardize", True):
            data_np = StandardScaler().fit_transform(data_np)

        return data_np, variable_names

    # --------------------------------------------------------------------- #
    # Individual algorithms
    # --------------------------------------------------------------------- #
    def _discover_with_pc(self, X: np.ndarray,
                          interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:

        if not (CAUSAL_LEARN_AVAILABLE and pc):
            logger.warning("PC not available – falling back to correlation.")
            return self._discover_with_correlation(X)

        try:
            cg = pc(X, alpha=self.significance_level, indep_test="fisherz",
                    stable=True, correction_name="BH")
            A = _graph_to_adjacency(cg.G, X.shape[1])
            return {
                "adjacency_matrix": A,
                "confidence_scores": np.ones_like(A) * 0.8,
            }
        except Exception as exc:  # pragma: no cover
            logger.error("PC failed: %s", exc)
            return self._discover_with_correlation(X)

    def _discover_with_ges(self, X: np.ndarray,
                           interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:

        if not (CAUSAL_LEARN_AVAILABLE and ges):
            logger.warning("GES not available – falling back to correlation.")
            return self._discover_with_correlation(X)

        try:
            record = ges(X, score_func="local_score_BIC")   # type: ignore[arg-type]
            A = _graph_to_adjacency(record["G"], X.shape[1])
            return {
                "adjacency_matrix": A,
                "confidence_scores": np.ones_like(A) * 0.8,
            }
        except Exception as exc:  # pragma: no cover
            logger.error("GES failed: %s", exc)
            return self._discover_with_correlation(X)

    def _discover_with_lingam(self, X: np.ndarray,
                              interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not (CAUSAL_LEARN_AVAILABLE and DirectLiNGAM):
            logger.warning("LiNGAM not available – falling back to correlation.")
            return self._discover_with_correlation(X)

        try:
            model = DirectLiNGAM()
            model.fit(X)
            A = np.asarray(model.adjacency_matrix_, dtype=float)
            return {
                "adjacency_matrix": A,
                "confidence_scores": np.abs(A),
            }
        except Exception as exc:  # pragma: no cover
            logger.error("LiNGAM failed: %s", exc)
            return self._discover_with_correlation(X)

    # ---- NOTEARS (simple NP implementation) ----------------------------- #
    def _discover_with_notears(self, X: np.ndarray,
                               interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        logger.info("Running NOTEARS …")
        try:
            A = self._notears_algorithm(X)
            conf = self._bootstrap_confidence(X, "notears")
            return {"adjacency_matrix": A, "confidence_scores": conf}
        except Exception as exc:  # pragma: no cover
            logger.error("NOTEARS failed: %s", exc)
            return self._discover_with_correlation(X)

    def _notears_algorithm(self, X: np.ndarray,
                           lambda_1: float = 0.1,
                           lr: float = 0.01,
                           max_iter: int = 100) -> np.ndarray:

        n, d = X.shape
        W = np.random.randn(d, d) * 0.1
        np.fill_diagonal(W, 0)

        def _h(Wm: np.ndarray) -> float:
            return np.trace(np.exp(Wm * Wm)) - d

        rho, alpha, h_tol = 1.0, 0.0, 1e-8

        for _ in range(max_iter):
            # Gradient of squared error
            grad = -X.T @ (X - X @ W) / n + lambda_1 * np.sign(W)
            # Gradient of acyclicity constraint
            E = np.exp(W * W)
            grad_h = 2 * W * E
            # Update
            W -= lr * (grad + (alpha + rho * _h(W)) * grad_h)
            np.fill_diagonal(W, 0)
            if _h(W) <= h_tol:
                break
        W[np.abs(W) < 0.05] = 0
        return W

    # ---- Bayesian network (pgmpy) --------------------------------------- #
    def _discover_with_bayesian_network(self, X: np.ndarray,
                                        interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not (PGMPY_AVAILABLE and HillClimbSearch and BicScore):
            logger.warning("pgmpy not available – falling back to correlation.")
            return self._discover_with_correlation(X)

        try:
            df = pd.DataFrame(X, columns=[f"Var_{i}" for i in range(X.shape[1])])
            hc = HillClimbSearch(df)
            model = hc.estimate(scoring_method=BicScore(df))
            A = _graph_to_adjacency(model, X.shape[1])
            return {"adjacency_matrix": A, "confidence_scores": np.ones_like(A) * 0.7}
        except Exception as exc:  # pragma: no cover
            logger.error("BayesianNet failed: %s", exc)
            return self._discover_with_correlation(X)

    # ---- Deep-learning toy baseline ------------------------------------- #
    def _discover_with_deep_learning(self, X: np.ndarray,
                                     interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not installed – using correlation fallback.")
            return self._discover_with_correlation(X)
        try:
            A = self._deep_causal_discovery(X)
            return {"adjacency_matrix": A, "confidence_scores": None}
        except Exception as exc:  # pragma: no cover
            logger.error("Deep discovery failed: %s", exc)
            return self._discover_with_correlation(X)

    def _deep_causal_discovery(self, X_np: np.ndarray) -> np.ndarray:
        """
        Very lightweight VAE-style adjacency-learning toy model.
        NOT intended for production research, but useful as a placeholder.
        """
        n, d = X_np.shape
        X = torch.tensor(X_np, dtype=torch.float32)

        class Net(nn.Module):
            def __init__(self, d_: int):
                super().__init__()
                self.logits = nn.Parameter(torch.randn(d_, d_))
                self.mech = nn.ModuleList([nn.Linear(d_, 1) for _ in range(d_)])

            def forward(self, x):
                A = torch.sigmoid(self.logits)
                outs = []
                for i in range(self.logits.shape[0]):
                    mask = A[i].clone()
                    mask[i] = 0.0
                    outs.append(self.mech[i](x * mask))
                return torch.cat(outs, dim=1), A

        net = Net(d)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)

        for epoch in range(500):
            opt.zero_grad()
            recon, A = net(X)
            loss = nn.MSELoss()(recon, X)
            loss += 0.05 * A.sum()
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                logger.debug("Epoch %04d | loss %.4f", epoch, loss.item())
        with torch.no_grad():
            _, A = net(X)
            A = A.numpy()
        A[np.abs(A) < 0.1] = 0
        return A

    # ---- Correlation (default fallback) --------------------------------- #
    def _discover_with_correlation(self, X: np.ndarray,
                                   interventions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        corr = np.corrcoef(X.T)
        thr = self.config.get("correlation_threshold", 0.3)
        A = (np.abs(corr) > thr).astype(float)
        np.fill_diagonal(A, 0)
        return {"adjacency_matrix": A,
                "confidence_scores": np.abs(corr),
                "correlation_matrix": corr}

    # --------------------------------------------------------------------- #
    # Bootstrapped edge-confidence
    # --------------------------------------------------------------------- #
    def _bootstrap_confidence(self, X: np.ndarray, method: str) -> np.ndarray:
        n, d = X.shape
        counts = np.zeros((d, d))
        for _ in range(self.bootstrap_samples):
            idx = np.random.choice(n, n, replace=True)
            boot = X[idx]
            if method == "notears":
                res = self._discover_with_notears(boot)
            elif method == "pc":
                res = self._discover_with_pc(boot)
            elif method == "ges":
                res = self._discover_with_ges(boot)
            else:
                res = self._discover_with_correlation(boot)
            A = res["adjacency_matrix"]
            counts += (A > 0).astype(int)
        return counts / self.bootstrap_samples

    # --------------------------------------------------------------------- #
    # Network analysis utilities
    # --------------------------------------------------------------------- #
    def _analyze_causal_structure(self, res: Dict[str, Any]) -> Dict[str, Any]:
        A = res["adjacency_matrix"]
        d = A.shape[1]
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)

        num_edges = int(np.sum(A > 0))
        density = num_edges / (d * (d - 1)) if d > 1 else 0.0
        is_dag = nx.is_directed_acyclic_graph(G)
        scc = list(nx.strongly_connected_components(G))

        # Hubs & authorities
        try:
            hubs, auths = nx.hits(G, max_iter=1000, normalized=True)
            hub_scores = [hubs.get(i, 0.0) for i in G.nodes()]
            auth_scores = [auths.get(i, 0.0) for i in G.nodes()]
        except (nx.exception.PowerIterationFailedConvergence, nx.NetworkXError):
            hub_scores = auth_scores = [0.0] * d

        return {
            "num_edges": num_edges,
            "network_density": density,
            "is_dag": is_dag,
            "num_strongly_connected_components": len(scc),
            "hub_scores": hub_scores,
            "authority_scores": auth_scores,
        }

# --------------------------------------------------------------------------- #
# Causal Inference Engine
# --------------------------------------------------------------------------- #
class CausalInferenceEngine:
    """Engine for causal inference and intervention prediction."""
    
    def __init__(self, causal_graph: np.ndarray, data: np.ndarray):
        self.causal_graph = causal_graph
        self.data = data
        self.structural_equations = {}
        self._fit_structural_equations()
    
    def _fit_structural_equations(self):
        """Fit structural equation models for each variable."""
        n_vars = self.causal_graph.shape[0]
        
        for i in range(n_vars):
            parents = np.where(self.causal_graph[:, i])[0]
            
            if len(parents) > 0:
                X = self.data[:, parents]
                y = self.data[:, i]
                
                # Fit structural equation (using regularized regression)
                model = LassoCV(cv=3, random_state=42)
                model.fit(X, y)
                
                self.structural_equations[i] = {
                    'model': model,
                    'parents': parents,
                    'noise_variance': np.var(y - model.predict(X))
                }
            else:
                # No parents, just noise
                self.structural_equations[i] = {
                    'model': None,
                    'parents': [],
                    'noise_variance': np.var(self.data[:, i])
                }
    
    def predict_intervention_effect(self, 
                                  intervention_targets: List[int],
                                  intervention_values: List[float],
                                  target_variables: Optional[List[int]] = None) -> Dict:
        """
        Predict the effect of interventions using do-calculus.
        
        Args:
            intervention_targets: Variables to intervene on
            intervention_values: Values to set for interventions
            target_variables: Variables to predict effects for
            
        Returns:
            Dictionary with predicted effects
        """
        if target_variables is None:
            target_variables = list(range(self.causal_graph.shape[0]))
        
        # Simulate intervention
        simulated_data = self._simulate_intervention(
            intervention_targets, intervention_values
        )
        
        # Compute effects
        effects = {}
        baseline_means = np.mean(self.data, axis=0)
        intervention_means = np.mean(simulated_data, axis=0)
        
        for var in target_variables:
            effect = intervention_means[var] - baseline_means[var]
            effects[var] = {
                'average_treatment_effect': effect,
                'baseline_mean': baseline_means[var],
                'intervention_mean': intervention_means[var],
                'effect_size': effect / np.std(self.data[:, var]) if np.std(self.data[:, var]) > 0 else 0
            }
        
        return {
            'effects': effects,
            'intervention_targets': intervention_targets,
            'intervention_values': intervention_values,
            'simulated_data': simulated_data
        }
    
    def _simulate_intervention(self, 
                             intervention_targets: List[int],
                             intervention_values: List[float],
                             n_samples: int = 1000) -> np.ndarray:
        """Simulate data under intervention using structural equations."""
        n_vars = self.causal_graph.shape[0]
        simulated_data = np.zeros((n_samples, n_vars))
        
        # Topological sort for causal ordering
        G = nx.from_numpy_array(self.causal_graph, create_using=nx.DiGraph)
        try:
            causal_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # If not DAG, use heuristic ordering
            causal_order = list(range(n_vars))
        
        # Generate samples following causal order
        for sample_idx in range(n_samples):
            for var in causal_order:
                if var in intervention_targets:
                    # Set intervention value
                    intervention_idx = intervention_targets.index(var)
                    simulated_data[sample_idx, var] = intervention_values[intervention_idx]
                else:
                    # Generate from structural equation
                    eq_info = self.structural_equations[var]
                    
                    if eq_info['model'] is not None and len(eq_info['parents']) > 0:
                        # Has parents
                        parent_values = simulated_data[sample_idx, eq_info['parents']]
                        predicted_value = eq_info['model'].predict(parent_values.reshape(1, -1))[0]
                        
                        # Add noise
                        noise = np.random.normal(0, np.sqrt(eq_info['noise_variance']))
                        simulated_data[sample_idx, var] = predicted_value + noise
                    else:
                        # No parents, sample from marginal distribution
                        marginal_mean = np.mean(self.data[:, var])
                        marginal_std = np.sqrt(eq_info['noise_variance'])
                        simulated_data[sample_idx, var] = np.random.normal(marginal_mean, marginal_std)
        
        return simulated_data

# --------------------------------------------------------------------------- #
# Evaluation and Analysis functions
# --------------------------------------------------------------------------- #
def run_causal_discovery(causal_factors: np.ndarray,
                        perturbation_labels: np.ndarray,
                        config: Dict) -> Dict[str, Any]:
    """
    Main function to run causal discovery analysis.
    """
    logger.info("SEARCH: Starting comprehensive causal discovery analysis...")
    discovery_engine = CausalDiscoveryEngine(config)
    
    # ... (rest of the function from the original file can be pasted here)
    # This is just a placeholder to show where it would go.
    # The logic for combining data and calling discover_causal_structure
    # would be needed.

def evaluate_discovery_performance(true_adjacency: np.ndarray,
                                   discovered_adjacency: np.ndarray) -> Dict[str, float]:
    """Evaluate causal discovery performance against ground truth."""
    true_edges = (true_adjacency > 0).flatten()
    discovered_edges = (discovered_adjacency > 0).flatten()
    
    tp = np.sum((true_edges == 1) & (discovered_edges == 1))
    fp = np.sum((true_edges == 0) & (discovered_edges == 1))
    fn = np.sum((true_edges == 1) & (discovered_edges == 0))
    tn = np.sum((true_edges == 0) & (discovered_edges == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'edge_precision': precision,
        'edge_recall': recall,
        'edge_f1': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }

# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def visualize_causal_network(discovery_results: Dict[str, Any], 
                           save_path: Optional[str] = None,
                           show_confidence: bool = True) -> None:
    """Visualize the discovered causal network."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("Matplotlib/Seaborn not available for visualization")
        return
        
    A = discovery_results['adjacency_matrix']
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    variable_names = discovery_results.get('variable_names', [])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, iterations=50)

    node_sizes = [300 + 100 * G.out_degree(n) for n in G.nodes]
    node_colors = [G.in_degree(n) for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                           node_color=node_colors, cmap='viridis', alpha=0.9)

    edge_weights = [A[u,v] for u,v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=[w*2 for w in edge_weights], 
                           alpha=0.6, arrowsize=15)
    
    if variable_names:
        labels = {i: name for i, name in enumerate(variable_names)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
        
    ax.set_title(f"Discovered Causal Network ({discovery_results['method']})")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show() 