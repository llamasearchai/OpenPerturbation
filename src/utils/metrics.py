def calculate_p_value(observed: float, baseline: np.ndarray) -> float:
    """Calculate empirical p-value"""
    if len(baseline) == 0:
        return 1.0
    
    # Handle NaN and infinite values
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) == 0:
        return 1.0
    
    # Calculate p-value
    return np.mean(baseline >= observed) 