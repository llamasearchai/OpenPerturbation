import numpy as np
from numpy.typing import NDArray
from typing import Union

def calculate_p_value(observed: float, baseline: NDArray) -> float:
    """Calculate empirical p-value"""
    if len(baseline) == 0:
        return 1.0
    
    # Handle NaN and infinite values
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) == 0:
        return 1.0
    
    # Calculate p-value and ensure float return type
    p_value = np.mean(baseline >= observed)
    return float(p_value)  # Explicit conversion to Python float 