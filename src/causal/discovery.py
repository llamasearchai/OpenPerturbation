"""
Causal Discovery Module for OpenPerturbation

Implements various causal discovery algorithms for perturbation biology.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# Import causal discovery libraries with fallbacks
try:
    from causallearn.search.ConstraintBased.PC import pc
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    warnings.warn("causal-learn not available")
    CAUSAL_LEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    warnings.warn("NetworkX not available") 
    NETWORKX_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("SciPy not available")
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_causal_discovery(
    causal_factors: np.ndarray,
    perturbation_labels: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run causal discovery analysis on perturbation data.
    
    Args:
        causal_factors: Input data matrix (n_samples x n_features)
        perturbation_labels: Perturbation conditions (n_samples x 1)
        config: Configuration dictionary with method and parameters
        
    Returns:
        Dictionary containing causal discovery results
    """
    method = config.get("method", "correlation")
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(causal_factors.shape[1])])
    alpha = config.get("alpha", 0.05)
    
    logger.info(f"Running causal discovery with method: {method}")
    
    if method == "pc" and CAUSAL_LEARN_AVAILABLE:
        return _run_pc_algorithm(causal_factors, config, variable_names)
    elif method == "correlation" or not CAUSAL_LEARN_AVAILABLE:
        return _run_correlation_based(causal_factors, config, variable_names, alpha)
    elif method == "granger":
        return _run_granger_causality(causal_factors, config, variable_names, alpha)
    else:
        # Fallback to correlation method
        logger.warning(f"Method {method} not available, falling back to correlation")
        return _run_correlation_based(causal_factors, config, variable_names, alpha)


def _run_pc_algorithm(
    data: np.ndarray, 
    config: Dict[str, Any], 
    variable_names: List[str]
) -> Dict[str, Any]:
    """Run PC algorithm for causal discovery."""
    try:
        alpha = config.get("alpha", 0.05)
        
        # Run PC algorithm
        cg = pc(data, alpha=alpha, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2)
        
        # Extract adjacency matrix
        adjacency_matrix = cg.G.graph
        
        # Convert to standard format
        n_vars = len(variable_names)
        adj_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i < adjacency_matrix.shape[0] and j < adjacency_matrix.shape[1]:
                    adj_matrix[i, j] = adjacency_matrix[i, j]
        
        return {
            "adjacency_matrix": adj_matrix.tolist(),
            "method": "pc",
            "variable_names": variable_names,
            "n_samples": data.shape[0],
            "n_variables": data.shape[1],
            "alpha": alpha,
            "message": "PC algorithm completed successfully",
            "causal_metrics": {
                "causal_network_density": np.mean(adj_matrix),
                "total_causal_edges": np.sum(adj_matrix > 0)
            }
        }
        
    except Exception as e:
        logger.error(f"PC algorithm failed: {e}")
        # Fallback to correlation
        return _run_correlation_based(data, config, variable_names, config.get("alpha", 0.05))


def _run_correlation_based(
    data: np.ndarray,
    config: Dict[str, Any],
    variable_names: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Run correlation-based causal discovery as fallback."""
    
    n_samples, n_vars = data.shape
    correlation_matrix = np.corrcoef(data.T)
    
    # Threshold correlation matrix to get causal adjacency
    correlation_threshold = config.get("correlation_threshold", 0.3)
    adjacency_matrix = (np.abs(correlation_matrix) > correlation_threshold).astype(float)
    
    # Remove self-loops
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Compute p-values if scipy available
    p_values = np.ones((n_vars, n_vars))
    if SCIPY_AVAILABLE:
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    try:
                        _, p_val = pearsonr(data[:, i], data[:, j])
                        p_values[i, j] = p_val
                    except Exception:
                        p_values[i, j] = 1.0
    
    # Apply significance threshold
    significant_edges = (p_values < alpha) & (adjacency_matrix > 0)
    final_adjacency = significant_edges.astype(float)
    
    return {
        "adjacency_matrix": final_adjacency.tolist(),
        "correlation_matrix": correlation_matrix.tolist(),
        "p_values": p_values.tolist(),
        "method": "correlation",
        "variable_names": variable_names,
        "n_samples": n_samples,
        "n_variables": n_vars,
        "alpha": alpha,
        "correlation_threshold": correlation_threshold,
        "message": "Correlation-based causal discovery completed",
        "causal_metrics": {
            "causal_network_density": np.mean(final_adjacency),
            "total_causal_edges": int(np.sum(final_adjacency)),
            "avg_correlation": float(np.mean(np.abs(correlation_matrix[final_adjacency > 0]))) if np.any(final_adjacency) else 0.0
        }
    }


def _run_granger_causality(
    data: np.ndarray,
    config: Dict[str, Any], 
    variable_names: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Run Granger causality analysis."""
    
    n_samples, n_vars = data.shape
    max_lag = config.get("max_lag", 5)
    
    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((n_vars, n_vars))
    p_values = np.ones((n_vars, n_vars))
    
    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, using correlation fallback")
        return _run_correlation_based(data, config, variable_names, alpha)
    
    try:
        # Simplified Granger causality using linear regression
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    try:
                        # Create lagged variables
                        y = data[max_lag:, j]  # target variable
                        X_restricted = np.column_stack([data[k:-max_lag+k, j] for k in range(max_lag)])  # lags of target only
                        X_full = np.column_stack([
                            X_restricted,
                            np.column_stack([data[k:-max_lag+k, i] for k in range(max_lag)])  # lags of predictor
                        ])
                        
                        if len(y) > max_lag + 1:
                            # Fit restricted model (only lags of target)
                            from sklearn.linear_model import LinearRegression
                            reg_restricted = LinearRegression().fit(X_restricted, y)
                            mse_restricted = np.mean((y - reg_restricted.predict(X_restricted))**2)
                            
                            # Fit full model (lags of target + predictor) 
                            reg_full = LinearRegression().fit(X_full, y)
                            mse_full = np.mean((y - reg_full.predict(X_full))**2)
                            
                            # F-test for significance
                            n = len(y)
                            p1 = X_restricted.shape[1]
                            p2 = X_full.shape[1]
                            
                            if mse_full > 0:
                                f_stat = ((mse_restricted - mse_full) / (p2 - p1)) / (mse_full / (n - p2))
                                p_val = 1 - stats.f.cdf(f_stat, p2 - p1, n - p2)
                                
                                p_values[i, j] = p_val
                                if p_val < alpha:
                                    adjacency_matrix[i, j] = 1.0
                    except Exception as e:
                        logger.debug(f"Granger causality failed for {i}->{j}: {e}")
                        continue
                        
    except ImportError:
        logger.warning("sklearn not available for Granger causality, using correlation fallback")
        return _run_correlation_based(data, config, variable_names, alpha)
    
    return {
        "adjacency_matrix": adjacency_matrix.tolist(),
        "p_values": p_values.tolist(),
        "method": "granger",
        "variable_names": variable_names,
        "n_samples": n_samples,
        "n_variables": n_vars,
        "alpha": alpha,
        "max_lag": max_lag,
        "message": "Granger causality analysis completed",
        "causal_metrics": {
            "causal_network_density": float(np.mean(adjacency_matrix)),
            "total_causal_edges": int(np.sum(adjacency_matrix)),
            "avg_p_value": float(np.mean(p_values[adjacency_matrix > 0])) if np.any(adjacency_matrix) else 1.0
        }
    }


def validate_causal_graph(
    adjacency_matrix: np.ndarray,
    variable_names: List[str],
    data: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Validate and analyze causal graph structure."""
    
    n_vars = len(variable_names)
    
    # Basic graph properties
    n_edges = np.sum(adjacency_matrix > 0)
    density = n_edges / (n_vars * (n_vars - 1)) if n_vars > 1 else 0
    
    # Check for cycles
    has_cycles = False
    if NETWORKX_AVAILABLE:
        try:
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
            has_cycles = not nx.is_directed_acyclic_graph(G)
        except Exception:
            has_cycles = False
    
    validation_results = {
        "n_variables": n_vars,
        "n_edges": int(n_edges),
        "density": float(density),
        "has_cycles": has_cycles,
        "is_valid_dag": not has_cycles,
        "variable_names": variable_names,
        "adjacency_shape": adjacency_matrix.shape
    }
    
    # Additional analysis if data provided
    if data is not None:
        try:
            # Compute edge strengths
            edge_strengths = []
            for i in range(n_vars):
                for j in range(n_vars):
                    if adjacency_matrix[i, j] > 0:
                        if SCIPY_AVAILABLE:
                            corr, _ = pearsonr(data[:, i], data[:, j])
                            edge_strengths.append(abs(corr))
                        else:
                            edge_strengths.append(1.0)
            
            validation_results.update({
                "avg_edge_strength": float(np.mean(edge_strengths)) if edge_strengths else 0.0,
                "edge_strength_std": float(np.std(edge_strengths)) if edge_strengths else 0.0,
                "data_samples": data.shape[0]
            })
        except Exception as e:
            logger.debug(f"Failed to compute edge strengths: {e}")
    
    return validation_results


def convert_adjacency_to_networkx(
    adjacency_matrix: np.ndarray,
    variable_names: List[str]
) -> Any:
    """Convert adjacency matrix to NetworkX graph."""
    
    if not NETWORKX_AVAILABLE:
        logger.warning("NetworkX not available")
        return None
    
    try:
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        
        # Add node labels
        node_mapping = {i: name for i, name in enumerate(variable_names)}
        G = nx.relabel_nodes(G, node_mapping)
        
        return G
    except Exception as e:
        logger.error(f"Failed to convert to NetworkX: {e}")
        return None


def compute_causal_effects(
    adjacency_matrix: np.ndarray,
    intervention_targets: List[int],
    variable_names: List[str]
) -> Dict[str, Any]:
    """Compute predicted causal effects of interventions."""
    
    n_vars = len(variable_names)
    
    # Initialize effect matrix
    effect_matrix = np.zeros((n_vars, n_vars))
    
    # Compute direct and indirect effects
    for target in intervention_targets:
        if target < n_vars:
            # Direct effects
            direct_effects = adjacency_matrix[target, :]
            effect_matrix[target, :] = direct_effects
            
            # Indirect effects (simplified - just 1-hop)
            for intermediate in range(n_vars):
                if adjacency_matrix[target, intermediate] > 0:
                    indirect_effects = adjacency_matrix[intermediate, :]
                    effect_matrix[target, :] += 0.5 * indirect_effects  # Damped indirect effect
    
    # Normalize effects
    effect_matrix = np.clip(effect_matrix, 0, 1)
    
    return {
        "effect_matrix": effect_matrix.tolist(),
        "intervention_targets": intervention_targets,
        "variable_names": variable_names,
        "total_affected_variables": int(np.sum(effect_matrix > 0)),
        "max_effect_strength": float(np.max(effect_matrix)),
        "avg_effect_strength": float(np.mean(effect_matrix[effect_matrix > 0])) if np.any(effect_matrix) else 0.0
    } 