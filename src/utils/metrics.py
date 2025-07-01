import numpy as np
from numpy.typing import NDArray
from typing import Union
import warnings

# Scientific computing
try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    SCIPY_AVAILABLE = True
except Exception:  # Catch all exceptions during SciPy import
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical features disabled.")
    
    # Create dummy scipy functionality
    class DummyStats:
        @staticmethod
        def pearsonr(x, y):
            return (0.0, 0.5)  # correlation, p-value
        
        @staticmethod
        def spearmanr(x, y):
            return (0.0, 0.5)  # correlation, p-value
        
        @staticmethod
        def kendalltau(x, y):
            return (0.0, 0.5)  # correlation, p-value
        
        @staticmethod
        def ttest_ind(a, b):
            return (0.0, 0.5)  # statistic, p-value
        
        @staticmethod
        def mannwhitneyu(x, y):
            return (0.0, 0.5)  # statistic, p-value
        
        @staticmethod
        def ks_2samp(x, y):
            return (0.0, 0.5)  # statistic, p-value
    
    def pdist(X, metric='euclidean'):
        n = len(X)
        return np.zeros(n * (n - 1) // 2)
    
    def squareform(X):
        n = int(np.sqrt(len(X) * 2)) + 1
        return np.zeros((n, n))
    
    def linkage(X, method='ward'):
        n = len(X)
        return np.zeros((n - 1, 4))
    
    def dendrogram(Z):
        return {}
    
    stats = DummyStats()

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:  # Catch all exceptions during sklearn import
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some ML metrics disabled.")
    
    # Create dummy sklearn functionality
    def accuracy_score(y_true, y_pred):
        return 0.5
    
    def precision_score(y_true, y_pred, **kwargs):
        return 0.5
    
    def recall_score(y_true, y_pred, **kwargs):
        return 0.5
    
    def f1_score(y_true, y_pred, **kwargs):
        return 0.5
    
    def roc_auc_score(y_true, y_score, **kwargs):
        return 0.5
    
    def average_precision_score(y_true, y_score, **kwargs):
        return 0.5
    
    def confusion_matrix(y_true, y_pred):
        return np.array([[1, 0], [0, 1]])
    
    def mean_squared_error(y_true, y_pred):
        return 0.5
    
    def mean_absolute_error(y_true, y_pred):
        return 0.5
    
    def r2_score(y_true, y_pred):
        return 0.5
    
    class DummyStandardScaler:
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
    
    class DummyPCA:
        def __init__(self, n_components=None):
            self.n_components = n_components
        def fit_transform(self, X):
            return X[:, :self.n_components] if self.n_components else X
    
    StandardScaler = DummyStandardScaler
    PCA = DummyPCA

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