"""
Feature Extraction Module for OpenPerturbation Platform

Comprehensive feature extraction for perturbation analysis supporting multiple data modalities.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd

# Runtime availability flags
HAS_SCIPY = False
HAS_SKIMAGE = False
HAS_SKLEARN = False
HAS_SCANPY = False

# Statistical analysis
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:  # Catch all exceptions during SciPy import
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some features may be limited.")

# Image processing - Handle all skimage imports with proper fallbacks
try:
    # Import skimage modules individually to handle import resolution
    from skimage import feature as skimage_feature  # type: ignore[import]
    from skimage import measure as skimage_measure   # type: ignore[import]
    from skimage import filters as skimage_filters   # type: ignore[import]
    from skimage import morphology as skimage_morphology  # type: ignore[import]
    from skimage import io as skimage_io  # type: ignore[import]
    
    # Import specific functions
    from skimage.feature import greycomatrix, greycoprops, blob_log  # type: ignore[import]
    from skimage.filters import threshold_otsu  # type: ignore[import]
    from skimage.measure import label, regionprops  # type: ignore[import]
    
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available. Image processing features disabled.")
    
    # Create fallback stubs for skimage functions
    def greycomatrix(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros((1, 1, 1, 1))
    
    def greycoprops(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros((1, 1))
    
    def blob_log(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.array([])
    
    def threshold_otsu(*args: Any, **kwargs: Any) -> float:
        return 0.5
    
    def label(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros((1, 1), dtype=int)
    
    def regionprops(*args: Any, **kwargs: Any) -> List[Any]:
        return []
    
    # Set skimage_io to None for type checking
    skimage_io = None

# Machine learning
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    HAS_SKLEARN = True
except Exception:  # Catch all exceptions during sklearn import
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. ML features may be limited.")

# Biological analysis
try:
    import scanpy as sc
    HAS_SCANPY = True
except Exception:  # Catch all exceptions during scanpy import
    HAS_SCANPY = False
    warnings.warn("scanpy not available. Single-cell analysis features disabled.")

# Define classes that may not be available
class OrganelleDetector:
    """Fallback organelle detector when specialized libraries are not available."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using fallback OrganelleDetector. Install specialized libraries for full functionality.")
    
    def detect_organelles(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic organelle detection using standard image processing."""
        if not HAS_SKIMAGE:
            return {"error": "scikit-image required for organelle detection"}
        
        try:
            # Basic blob detection as fallback
            blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
            
            return {
                "organelles": len(blobs),
                "positions": blobs.tolist() if len(blobs) > 0 else [],
                "method": "basic_blob_detection"
            }
        except Exception as e:
            return {"error": f"Organelle detection failed: {e}"}

class CellSegmenter:
    """Fallback cell segmenter when specialized libraries are not available."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using fallback CellSegmenter. Install specialized libraries for full functionality.")
    
    def segment_cells(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic cell segmentation using standard image processing."""
        if not HAS_SKIMAGE:
            return {"error": "scikit-image required for cell segmentation"}
        
        try:
            # Basic thresholding and labeling
            thresh = threshold_otsu(image)
            binary = image > thresh
            labeled = label(binary)
            props = regionprops(labeled)
            
            return {
                "cell_count": len(props),
                "areas": [prop.area for prop in props] if props else [],
                "centroids": [list(prop.centroid) for prop in props] if props else [],
                "method": "basic_thresholding"
            }
        except Exception as e:
            return {"error": f"Cell segmentation failed: {e}"}

class FeatureExtractor:
    """
    Comprehensive feature extraction for perturbation analysis.
    
    Supports multiple data modalities:
    - Single-cell RNA-seq data
    - Microscopy images
    - Multimodal fusion
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors based on available libraries
        if HAS_SKIMAGE:
            self.organelle_detector: Optional[OrganelleDetector] = OrganelleDetector()
            self.cell_segmenter: Optional[CellSegmenter] = CellSegmenter()
        else:
            self.organelle_detector = None
            self.cell_segmenter = None
            
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
    def extract_transcriptomic_features(self, 
                                      data: Union[pd.DataFrame, np.ndarray],
                                      gene_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract features from transcriptomic data.
        
        Args:
            data: Gene expression matrix (cells x genes or genes x cells)
            gene_names: List of gene names
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            if isinstance(data, pd.DataFrame):
                expression_matrix = data.values
                genes = gene_names or data.columns.tolist()
            else:
                expression_matrix = np.array(data)
                genes = gene_names or [f"gene_{i}" for i in range(expression_matrix.shape[1])]
            
            features = {
                "basic_stats": self._compute_basic_stats(expression_matrix),
                "gene_count": len(genes),
                "cell_count": expression_matrix.shape[0],
                "sparsity": self._compute_sparsity(expression_matrix),
            }
            
            # Add advanced features if libraries are available
            if HAS_SKLEARN:
                pca_result = self._compute_pca_features(expression_matrix)
                features["pca_features"] = pca_result
                
                hvg_result = self._find_highly_variable_genes(expression_matrix, genes)
                features["highly_variable_genes"] = hvg_result
            
            if HAS_SCANPY:
                scanpy_result = self._compute_scanpy_metrics(expression_matrix, genes)
                features["scanpy_metrics"] = scanpy_result
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting transcriptomic features: {e}")
            return {"error": str(e)}
    
    def extract_morphological_features(self, 
                                     image: np.ndarray,
                                     extract_organelles: bool = True) -> Dict[str, Any]:
        """
        Extract morphological features from microscopy images.
        
        Args:
            image: Input microscopy image
            extract_organelles: Whether to extract organelle features
            
        Returns:
            Dictionary containing morphological features
        """
        if not HAS_SKIMAGE:
            return {"error": "scikit-image required for morphological feature extraction"}
        
        try:
            features = {
                "image_shape": image.shape,
                "basic_stats": {
                    "mean_intensity": float(np.mean(image)),
                    "std_intensity": float(np.std(image)),
                    "min_intensity": float(np.min(image)),
                    "max_intensity": float(np.max(image))
                }
            }
            
            # Add texture features
            texture_features = self._compute_texture_features(image)
            if "error" not in texture_features:
                features["texture"] = texture_features
            
            # Add cell segmentation features
            if self.cell_segmenter:
                cell_features = self.cell_segmenter.segment_cells(image)
                if "error" not in cell_features:
                    features["cell_segmentation"] = cell_features
            
            # Add organelle detection if requested
            if extract_organelles and self.organelle_detector:
                organelle_features = self.organelle_detector.detect_organelles(image)
                if "error" not in organelle_features:
                    features["organelles"] = organelle_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting morphological features: {e}")
            return {"error": str(e)}
    
    def extract_multimodal_features(self,
                                  transcriptomic_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                                  image_data: Optional[np.ndarray] = None,
                                  **kwargs: Any) -> Dict[str, Any]:
        """
        Extract features from multiple modalities.
        
        Args:
            transcriptomic_data: Gene expression data
            image_data: Microscopy image data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing multimodal features
        """
        try:
            features: Dict[str, Any] = {}
            
            # Extract transcriptomic features
            if transcriptomic_data is not None:
                trans_features = self.extract_transcriptomic_features(transcriptomic_data)
                features["transcriptomic"] = trans_features
            
            # Extract morphological features
            if image_data is not None:
                morph_features = self.extract_morphological_features(image_data)
                features["morphological"] = morph_features
            
            # Extract cross-modal features if both modalities are available
            if transcriptomic_data is not None and image_data is not None:
                # Convert transcriptomic data to numpy array if needed
                if isinstance(transcriptomic_data, pd.DataFrame):
                    trans_array = transcriptomic_data.values
                else:
                    trans_array = np.array(transcriptomic_data)
                
                cross_modal_features = self._compute_cross_modal_features(trans_array, image_data)
                features["cross_modal"] = cross_modal_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting multimodal features: {e}")
            return {"error": str(e)}
    
    def _compute_basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Compute basic statistical features."""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data.flatten())) if HAS_SCIPY else 0.0,
            "kurtosis": float(stats.kurtosis(data.flatten())) if HAS_SCIPY else 0.0
        }
    
    def _compute_sparsity(self, data: np.ndarray) -> float:
        """Compute sparsity of the data matrix."""
        return float(np.sum(data == 0) / data.size)
    
    def _compute_pca_features(self, data: np.ndarray, n_components: int = 10) -> Dict[str, Any]:
        """Compute PCA features."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn required for PCA"}
        
        try:
            pca = PCA(n_components=min(n_components, min(data.shape)))
            pca_result = pca.fit_transform(data)
            
            return {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "n_components": pca.n_components_,
                "components_shape": pca_result.shape
            }
        except Exception as e:
            return {"error": f"PCA computation failed: {e}"}
    
    def _find_highly_variable_genes(self, data: np.ndarray, gene_names: List[str]) -> Dict[str, Any]:
        """Find highly variable genes."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn required for gene selection"}
        
        try:
            # Create dummy labels for feature selection
            y_dummy = np.random.randint(0, 2, size=data.shape[0])
            
            selector = SelectKBest(f_classif, k=min(100, data.shape[1]))
            selector.fit(data, y_dummy)
            
            selected_indices = selector.get_support(indices=True)
            selected_genes = [gene_names[i] for i in (selected_indices or []) if i < len(gene_names)]
            
            return {
                "highly_variable_genes": selected_genes,
                "n_selected": len(selected_genes),
                "selection_scores": selector.scores_[selected_indices].tolist() if selector.scores_ is not None else []
            }
        except Exception as e:
            return {"error": f"Gene selection failed: {e}"}
    
    def _compute_scanpy_metrics(self, data: np.ndarray, gene_names: List[str]) -> Dict[str, Any]:
        """Compute scanpy-based metrics."""
        if not HAS_SCANPY:
            return {"error": "scanpy required for single-cell metrics"}
        
        try:
            # Create AnnData object
            import anndata as ad
            adata = ad.AnnData(X=data)
            
            # Ensure gene_names length matches data dimensions
            n_genes = min(len(gene_names), data.shape[1])
            if n_genes > 0:
                adata.var_names = gene_names[:n_genes]
            
            # Basic QC metrics
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            
            return {
                "n_genes_by_counts": float(adata.obs['n_genes_by_counts'].mean()),
                "total_counts": float(adata.obs['total_counts'].mean()),
                "pct_counts_in_top_50_genes": float(adata.obs['pct_counts_in_top_50_genes'].mean())
            }
        except Exception as e:
            return {"error": f"Scanpy metrics computation failed: {e}"}
    
    def _compute_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Compute texture features from grayscale image."""
        if not HAS_SKIMAGE:
            return {"error": "scikit-image required for texture features"}
        
        try:
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # GLCM features
            glcm = greycomatrix(image_uint8, [1], [0], symmetric=True, normed=True)
            
            return {
                "contrast": float(greycoprops(glcm, 'contrast')[0, 0]),
                "dissimilarity": float(greycoprops(glcm, 'dissimilarity')[0, 0]),
                "homogeneity": float(greycoprops(glcm, 'homogeneity')[0, 0]),
                "energy": float(greycoprops(glcm, 'energy')[0, 0]),
                "correlation": float(greycoprops(glcm, 'correlation')[0, 0])
            }
        except Exception as e:
            return {"error": f"Texture feature computation failed: {e}"}
    
    def _compute_cross_modal_features(self, 
                                    transcriptomic_data: Optional[np.ndarray],
                                    image_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Compute cross-modal features."""
        try:
            features: Dict[str, Any] = {"computed": True}
            
            if transcriptomic_data is not None and image_data is not None:
                # Simple correlation between modalities
                if HAS_SCIPY:
                    # Flatten and compute correlation
                    trans_flat = transcriptomic_data.flatten()[:1000]  # Sample for efficiency
                    image_flat = image_data.flatten()[:1000]
                    
                    min_length = min(len(trans_flat), len(image_flat))
                    if min_length > 0:
                        trans_sample = trans_flat[:min_length]
                        image_sample = image_flat[:min_length]
                        
                        # Pyright: SciPy returns a specialised object in >=1.11; type stubs lag behind
                        correlation_result = stats.pearsonr(  # type: ignore[arg-type]
                            trans_sample.astype(np.float64),
                            image_sample.astype(np.float64)
                        )
                        # Extract correlation coefficient safely
                        if hasattr(correlation_result, 'correlation'):
                            # New scipy version returns object with .correlation attribute
                            correlation_value = float(correlation_result.correlation)
                        else:
                            # Older scipy version returns tuple
                            correlation_value = float(correlation_result[0])
                        
                        features["cross_modal_correlation"] = correlation_value
            
            return features
            
        except Exception as e:
            return {"error": f"Cross-modal feature computation failed: {e}"}

def extract_features_from_file(file_path: Union[str, Path], 
                             feature_type: str = "auto") -> Dict[str, Any]:
    """
    Extract features from a file.
    
    Args:
        file_path: Path to the data file
        feature_type: Type of features to extract ("transcriptomic", "morphological", "auto")
        
    Returns:
        Dictionary containing extracted features
    """
    extractor = FeatureExtractor()
    file_path = Path(file_path)
    
    try:
        if feature_type == "auto":
            # Determine feature type from file extension
            if file_path.suffix.lower() in ['.csv', '.tsv', '.txt', '.h5', '.h5ad']:
                feature_type = "transcriptomic"
            elif file_path.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                feature_type = "morphological"
            else:
                return {"error": f"Cannot determine feature type for {file_path.suffix}"}
        
        if feature_type == "transcriptomic":
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path, index_col=0)
                return extractor.extract_transcriptomic_features(data)
            else:
                return {"error": f"Unsupported transcriptomic file format: {file_path.suffix}"}
        
        elif feature_type == "morphological":
            if HAS_SKIMAGE and skimage_io is not None:
                image = skimage_io.imread(file_path)
                return extractor.extract_morphological_features(image)
            else:
                return {"error": "scikit-image required for morphological feature extraction"}
        
        else:
            return {"error": f"Unknown feature type: {feature_type}"}
            
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}

# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    extractor = FeatureExtractor()
    
    # Test transcriptomic features
    print("Testing transcriptomic feature extraction...")
    synthetic_expression = np.random.randn(100, 50)  # 100 cells, 50 genes
    gene_names = [f"GENE_{i}" for i in range(50)]
    
    trans_features = extractor.extract_transcriptomic_features(
        synthetic_expression, gene_names
    )
    print(f"Transcriptomic features extracted: {list(trans_features.keys())}")
    
    # Test morphological features if available
    if HAS_SKIMAGE:
        print("Testing morphological feature extraction...")
        synthetic_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        morph_features = extractor.extract_morphological_features(synthetic_image)
        print(f"Morphological features extracted: {list(morph_features.keys())}")
    
    print("Feature extraction testing completed!")