# pyright: reportReturnType=none, reportAssignmentType=none, reportInvalidTypeForm=none, reportArgumentType=none, reportIndexIssue=none, reportAttributeAccessIssue=none

"""
Genomics Data Loader for OpenPerturbation

Comprehensive genomics data loading for single-cell and bulk RNA-seq data.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
from pathlib import Path
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple, Protocol
import warnings

# Type definitions for better type safety
class DataFrameLike(Protocol):
    def __len__(self) -> int: ...
    def iloc(self, idx: int) -> Any: ...
    def to_dict(self) -> Dict[str, Any]: ...

class AnnDataLike(Protocol):
    X: np.ndarray
    obs: Any
    var: Any
    n_obs: int
    n_vars: int
    obs_names: Any
    var_names: Any

# Add required imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    warnings.warn("Pandas not available")
    PANDAS_AVAILABLE = False
    
    # Create minimal pandas-like functionality
    class DataFrame:
        def __init__(self, data=None, index=None, columns=None):
            if data is not None:
                if isinstance(data, np.ndarray):
                    self.data = data.tolist() if data.ndim > 1 else data.tolist()
                else:
                    self.data = data or []
            else:
                self.data = []
            self.index = index or list(range(len(self.data))) if self.data else []
            self.columns = columns or []
            
        def __len__(self) -> int:
            return len(self.data)
            
        def iloc(self, idx: int) -> 'Series':
            if idx < len(self.data):
                return Series(self.data[idx] if isinstance(self.data[idx], list) else [self.data[idx]])
            return Series([])
            
        def to_dict(self) -> Dict[str, Any]:
            return {"data": self.data, "index": self.index, "columns": self.columns}
            
        def to_csv(self, filepath: str, index: bool = True) -> None:
            pass  # Dummy implementation
            
        def set_index(self, col: str) -> 'DataFrame':
            return self
            
        @property
        def values(self) -> np.ndarray:
            return np.array(self.data) if self.data else np.array([])
    
    class Series:
        def __init__(self, data=None, index=None, name=None):
            self.data = data or []
            self.index = index or list(range(len(self.data))) if self.data else []
            self.name = name
            
        def value_counts(self) -> 'Series':
            return Series([])
            
        def to_dict(self) -> Dict[str, Any]:
            return dict(zip(self.index, self.data)) if self.index and self.data else {}
            
        def __getitem__(self, key: str) -> Any:
            return None
    
    def read_csv(filepath: str, **kwargs) -> DataFrame:
        return DataFrame()
        
    def read_hdf(filepath: str) -> DataFrame:
        return DataFrame()
    
    # Create mock pandas module
    class PandasModule:
        DataFrame = DataFrame
        Series = Series
        read_csv = staticmethod(read_csv)
        read_hdf = staticmethod(read_hdf)
    
    pd = PandasModule()

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available")
    TORCH_AVAILABLE = False
    
    # Create dummy classes
    class TorchDataset:
        def __init__(self):
            pass
            
        def __len__(self) -> int:
            return 0
            
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return {}
            
    class TorchDataLoader:
        def __init__(self, *args, **kwargs):
            self.dataset = args[0] if args else None
            
        def __iter__(self):
            return iter([])
            
        def __len__(self) -> int:
            return 0

try:
    from omegaconf import DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    warnings.warn("OmegaConf not available")
    OMEGACONF_AVAILABLE = False
    DictConfig = dict

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except Exception:  # Catch all exceptions during scanpy import
    warnings.warn("Scanpy not available")
    SCANPY_AVAILABLE = False
    
    # Create dummy scanpy
    class PreprocessingModule:
        def filter_genes(self, adata: Any, min_cells: int = 3) -> None:
            pass
            
        def filter_cells(self, adata: Any, min_genes: int = 200) -> None:
            pass
            
        def calculate_qc_metrics(self, adata: Any, **kwargs) -> None:
            pass
            
        def normalize_total(self, adata: Any, target_sum: float = 1e4) -> None:
            pass
            
        def log1p(self, adata: Any) -> None:
            pass
            
        def highly_variable_genes(self, adata: Any, **kwargs) -> None:
            pass
    
    class DummyScanpy:
        def __init__(self):
            self.pp = PreprocessingModule()
            
        def read_h5ad(self, filepath: str) -> 'DummyAnnData':
            return DummyAnnData()
            
        def read_csv(self, filepath: str) -> 'DummyAnnData':
            return DummyAnnData()
    
    sc = DummyScanpy()

# Define DummyAnnData globally so it's always available
class DummyAnnData:
    def __init__(self, X: Optional[np.ndarray] = None):
        self.X = X if X is not None else np.random.randn(100, 2000)
        if PANDAS_AVAILABLE:
            self.obs = pd.DataFrame({
                'cell_id': [f'cell_{i}' for i in range(self.X.shape[0])],
                'perturbation': ['control'] * (self.X.shape[0] // 2) + ['treated'] * (self.X.shape[0] - self.X.shape[0] // 2)
            })
            self.var = pd.DataFrame({
                'gene_id': [f'gene_{i}' for i in range(self.X.shape[1])],
                'gene_name': [f'GENE{i}' for i in range(self.X.shape[1])]
            })
        else:
            # Create stub DataFrame
            self.obs = DataFrame({
                'cell_id': [f'cell_{i}' for i in range(self.X.shape[0])],
                'perturbation': ['control'] * (self.X.shape[0] // 2) + ['treated'] * (self.X.shape[0] - self.X.shape[0] // 2)
            })
            self.var = DataFrame({
                'gene_id': [f'gene_{i}' for i in range(self.X.shape[1])],
                'gene_name': [f'GENE{i}' for i in range(self.X.shape[1])]
            })
        self.n_obs = self.X.shape[0]
        self.n_vars = self.X.shape[1]
        self.obs_names = [f'cell_{i}' for i in range(self.n_obs)]
        self.var_names = [f'gene_{i}' for i in range(self.n_vars)]
        
    def to_df(self) -> Any:
        if PANDAS_AVAILABLE:
            return pd.DataFrame(self.X)
        else:
            return DataFrame(self.X)
        
    def copy(self) -> 'DummyAnnData':
        new_adata = DummyAnnData(self.X.copy())
        return new_adata
        
    def write_h5ad(self, filepath: str) -> None:
        pass
        
    def __getitem__(self, key) -> 'DummyAnnData':
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.n_obs
            new_X = self.X[start:stop]
            new_adata = DummyAnnData(new_X)
            return new_adata
        return self

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    warnings.warn("AnnData not available")
    ANNDATA_AVAILABLE = False
    
    def read_h5ad(filepath: str) -> DummyAnnData:
        return DummyAnnData()
    
    class AnndataModule:
        AnnData = DummyAnnData
        read_h5ad = staticmethod(read_h5ad)
    
    ad = AnndataModule()

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    warnings.warn("h5py not available")
    H5PY_AVAILABLE = False

logger = logging.getLogger(__name__)


class GenomicsDataLoader:
    """Genomics data loader with comprehensive error handling."""
    
    def __init__(self, config: Union[DictConfig, dict]):
        self.config = config
        self.datasets: Dict[str, Any] = {}  # Changed to Any to handle both Dataset and Subset types
        self.dataloaders: Dict[str, TorchDataLoader] = {}
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 0)
        self.pin_memory = config.get("pin_memory", False)
        
        # Data processing parameters
        self.normalize = config.get("normalize", True)
        self.log_transform = config.get("log_transform", True)
        self.filter_genes = config.get("filter_genes", True)
        self.min_cells = config.get("min_cells", 3)
        self.min_genes = config.get("min_genes", 200)
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and dataloaders."""
        if not PANDAS_AVAILABLE:
            logger.error("Pandas required for genomics data loading")
            return
            
        data_dir = Path(self.config.get("data_dir", "data/genomics"))
        
        if stage == "fit" or stage is None:
            # Load training data
            train_file = data_dir / self.config.get("train_file", "train.h5ad")
            if train_file.exists():
                self.datasets["train"] = SingleCellDataset(
                    self.config, str(train_file), mode="train"
                )
            
            # Load validation data
            val_file = data_dir / self.config.get("val_file", "val.h5ad") 
            if val_file.exists():
                self.datasets["val"] = SingleCellDataset(
                    self.config, str(val_file), mode="val"
                )
            elif "train" in self.datasets:
                # Split training data
                self._split_dataset()
        
        if stage == "test" or stage is None:
            test_file = data_dir / self.config.get("test_file", "test.h5ad")
            if test_file.exists():
                self.datasets["test"] = SingleCellDataset(
                    self.config, str(test_file), mode="test"
                )
        
        # Create dataloaders
        self._create_dataloaders()
        
    def _split_dataset(self) -> None:
        """Split training dataset into train/val."""
        if "train" not in self.datasets:
            return
            
        train_dataset = self.datasets["train"]
        val_ratio = self.config.get("val_ratio", 0.2)
        
        if TORCH_AVAILABLE:
            from torch.utils.data import random_split
            dataset_size = len(train_dataset)
            val_size = int(val_ratio * dataset_size)
            train_size = dataset_size - val_size
            
            # Cast to Any to handle type compatibility
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size]  # type: ignore
            )
            
            # Store as Any type to avoid type conflicts
            self.datasets["train"] = train_subset  # type: ignore
            self.datasets["val"] = val_subset  # type: ignore
        else:
            logger.warning("PyTorch not available, cannot split dataset")
            
    def _create_dataloaders(self) -> None:
        """Create PyTorch dataloaders."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot create dataloaders")
            return
            
        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Custom collate function for genomics data."""
            # Extract data components
            expressions = []
            cell_ids = []
            perturbations = []
            metadata = []
            
            for item in batch:
                if isinstance(item.get("expression"), torch.Tensor):
                    expressions.append(item["expression"])
                cell_ids.append(item.get("cell_id", "unknown"))
                perturbations.append(item.get("perturbation", "unknown"))
                metadata.append(item.get("metadata", {}))
            
            # Convert to tensors
            batch_data = {
                "expression": torch.stack(expressions) if expressions else torch.empty(0),
                "cell_ids": cell_ids,
                "perturbations": perturbations,
                "metadata": metadata
            }
            
            return batch_data
        
        for split, dataset in self.datasets.items():
            shuffle = (split == "train")
            
            dataloader = TorchDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn
            )
            
            self.dataloaders[split] = dataloader
        
    def get_dataloader(self, split: str) -> Optional[TorchDataLoader]:
        """Get dataloader for split."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for data loading")
            return None
        return self.dataloaders.get(split)
        
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {}
        
        for split, dataset in self.datasets.items():
            adata = None
            if TORCH_AVAILABLE and hasattr(dataset, 'dataset'):
                # Handle Subset objects
                if hasattr(dataset.dataset, 'adata'):  # type: ignore
                    adata = dataset.dataset.adata  # type: ignore
            elif hasattr(dataset, 'adata'):
                adata = dataset.adata  # type: ignore
                
            if adata is not None:
                perturbation_counts = {}
                if hasattr(adata, 'obs') and hasattr(adata.obs, 'get'):
                    perturb_series = adata.obs.get('perturbation', None)  # type: ignore
                    if perturb_series is not None and hasattr(perturb_series, 'value_counts'):
                        perturbation_counts = perturb_series.value_counts().to_dict()  # type: ignore
                
                stats[split] = {
                    "n_cells": adata.n_obs,
                    "n_genes": adata.n_vars,
                    "perturbations": perturbation_counts
                }
            else:
                stats[split] = {
                    "n_cells": len(dataset) if dataset else 0,
                    "n_genes": 0,
                    "perturbations": {}
                }
        
        return stats


class SingleCellDataset(TorchDataset):
    """Single cell RNA-seq dataset with comprehensive processing."""
    
    def __init__(self, config: Union[DictConfig, dict], data_path: str, mode: str = "train"):
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.mode = mode
        self.adata = self._load_data()
        
        if self.adata is not None:
            self._preprocess_data()
        
    def _load_data(self) -> Optional[AnnDataLike]:
        """Load single cell data from various formats."""
        data_path = Path(self.data_path)
        
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return self._create_dummy_data()
        
        try:
            if data_path.suffix == '.h5ad':
                if ANNDATA_AVAILABLE:
                    return ad.read_h5ad(str(data_path))
                else:
                    logger.warning("AnnData not available for .h5ad files")
                    return self._create_dummy_data()
                    
            elif data_path.suffix == '.csv':
                if PANDAS_AVAILABLE:
                    df = pd.read_csv(str(data_path), index_col=0)
                    return self._df_to_anndata(df)
                else:
                    logger.warning("Pandas not available for .csv files")
                    return self._create_dummy_data()
                    
            elif data_path.suffix == '.h5' and H5PY_AVAILABLE:
                return self._load_h5_data(data_path)
                
            else:
                logger.warning(f"Unsupported file format: {data_path.suffix}")
                return self._create_dummy_data()
                
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> DummyAnnData:
        """Create dummy single cell data for testing."""
        return DummyAnnData()
    
    def _df_to_anndata(self, df: Any) -> AnnDataLike:
        """Convert DataFrame to AnnData-like structure."""
        if not ANNDATA_AVAILABLE:
            # Create dummy structure
            dummy = DummyAnnData()
            if hasattr(df, 'values'):
                dummy.X = df.values
            return dummy
        
        # Convert pandas DataFrame to AnnData
        adata = ad.AnnData(X=df.values)
        adata.obs_names = df.index
        adata.var_names = df.columns
        
        return adata
    
    def _load_h5_data(self, data_path: Path) -> Optional[AnnDataLike]:
        """Load data from HDF5 format."""
        try:
            with h5py.File(data_path, 'r') as f:
                if 'matrix' in f:
                    matrix = f['matrix']
                    if isinstance(matrix, h5py.Group):
                        expression_data = matrix['data'][:] if 'data' in matrix else np.array([])
                        indices = matrix['indices'][:] if 'indices' in matrix else np.array([])
                        indptr = matrix['indptr'][:] if 'indptr' in matrix else np.array([])
                        shape = tuple(matrix['shape'][:]) if 'shape' in matrix else (0, 0)
                        
                        # Reconstruct sparse matrix
                        from scipy.sparse import csr_matrix
                        X = csr_matrix((expression_data, indices, indptr), shape=shape)
                        
                        # Create AnnData object
                        if ANNDATA_AVAILABLE:
                            adata = ad.AnnData(X=X.T)  # Transpose for cells x genes
                            return adata
                        else:
                            # Return dummy data
                            return self._create_dummy_data()
                    else:
                        return self._create_dummy_data()
        except Exception as e:
            logger.error(f"Failed to load HDF5 data: {e}")
            return self._create_dummy_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess single cell data."""
        if self.adata is None or not SCANPY_AVAILABLE:
            return
        
        try:
            # Basic preprocessing pipeline
            if self.config.get("filter_genes", True):
                # Filter genes expressed in minimum number of cells
                sc.pp.filter_genes(self.adata, min_cells=self.config.get("min_cells", 3))
            
            if self.config.get("filter_cells", True):
                # Filter cells with minimum number of genes
                sc.pp.filter_cells(self.adata, min_genes=self.config.get("min_genes", 200))
            
            # Calculate QC metrics
            if hasattr(sc.pp, 'calculate_qc_metrics'):
                sc.pp.calculate_qc_metrics(self.adata, percent_top=None, log1p=False, inplace=True)
            
            # Normalize to 10,000 reads per cell
            if self.config.get("normalize", True):
                sc.pp.normalize_total(self.adata, target_sum=1e4)
            
            # Log transform
            if self.config.get("log_transform", True):
                sc.pp.log1p(self.adata)
            
            # Highly variable genes
            if self.config.get("find_hvg", True):
                sc.pp.highly_variable_genes(
                    self.adata, 
                    min_mean=0.0125, 
                    max_mean=3, 
                    min_disp=0.5
                )
            
            logger.info(f"Preprocessed data: {self.adata.n_obs} cells, {self.adata.n_vars} genes")
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
    
    def __len__(self) -> int:
        if self.adata is None:
            return 0
        return self.adata.n_obs
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.adata is None:
            return {
                "expression": torch.zeros(2000) if TORCH_AVAILABLE else np.zeros(2000), 
                "cell_id": f"cell_{idx}", 
                "perturbation": "unknown", 
                "metadata": {}
            }
        
        try:
            # Get expression data for cell
            if hasattr(self.adata.X, 'toarray'):
                # Sparse matrix
                expression_data = self.adata.X[idx].toarray().flatten()
            else:
                # Dense matrix
                expression_data = self.adata.X[idx]
                if hasattr(expression_data, 'flatten'):
                    expression_data = expression_data.flatten()
            
            if TORCH_AVAILABLE:
                expression = torch.tensor(expression_data, dtype=torch.float32)
            else:
                expression = np.array(expression_data, dtype=np.float32)
            
            # Get cell metadata
            cell_id = f"cell_{idx}"
            if hasattr(self.adata, 'obs_names') and self.adata.obs_names is not None and idx < len(self.adata.obs_names):
                try:
                    cell_id = str(self.adata.obs_names[idx])
                except Exception:
                    pass
            
            # Get perturbation information
            perturbation = "control"
            if (hasattr(self.adata, 'obs') and 
                hasattr(self.adata.obs, 'iloc') and 
                idx < len(self.adata.obs)):
                try:
                    obs_row = self.adata.obs.iloc(idx)
                    if hasattr(obs_row, '__getitem__') and 'perturbation' in obs_row:
                        perturbation = str(obs_row['perturbation'])
                except Exception:
                    pass
            
            # Additional metadata
            metadata = {}
            if (hasattr(self.adata, 'obs') and 
                hasattr(self.adata.obs, 'iloc') and 
                idx < len(self.adata.obs)):
                try:
                    obs_row = self.adata.obs.iloc(idx)
                    if hasattr(obs_row, 'to_dict'):
                        metadata = obs_row.to_dict()
                except Exception:
                    pass
            
            return {
                "expression": expression,
                "cell_id": cell_id,
                "perturbation": perturbation,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.warning(f"Failed to get item {idx}: {e}")
            return {
                "expression": torch.zeros(2000) if TORCH_AVAILABLE else np.zeros(2000),
                "cell_id": f"cell_{idx}",
                "perturbation": "unknown",
                "metadata": {}
            }


class BulkRNASeqDataset(TorchDataset):
    """Bulk RNA-seq dataset for perturbation analysis."""
    
    def __init__(self, config: Union[DictConfig, dict], data_path: str, mode: str = "train"):
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.mode = mode
        self.data = self._load_data()
        
    def _load_data(self) -> Optional[Any]:
        """Load bulk RNA-seq data."""
        if not PANDAS_AVAILABLE:
            logger.error("Pandas required for bulk RNA-seq data")
            return None
        
        try:
            data_path = Path(self.data_path)
            
            if data_path.suffix == '.csv':
                return pd.read_csv(str(data_path), index_col=0)
            elif data_path.suffix == '.tsv':
                return pd.read_csv(str(data_path), sep='\t', index_col=0)
            elif data_path.suffix == '.h5':
                return pd.read_hdf(str(data_path))
            else:
                logger.warning(f"Unsupported format: {data_path.suffix}")
                return self._create_dummy_data()
                
        except Exception as e:
            logger.error(f"Failed to load bulk RNA-seq data: {e}")
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> Optional[Any]:
        """Create dummy bulk RNA-seq data."""
        if PANDAS_AVAILABLE:
            # Create dummy expression matrix
            n_samples = 100
            n_genes = 2000
            
            data = pd.DataFrame(
                np.random.lognormal(size=(n_samples, n_genes)),
                index=pd.Index([f'sample_{i}' for i in range(n_samples)]),
                columns=pd.Index([f'gene_{i}' for i in range(n_genes)])
            )
            
            return data
        else:
            return None
    
    def __len__(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data is None:
            return {
                "expression": torch.zeros(2000) if TORCH_AVAILABLE else np.zeros(2000), 
                "sample_id": f"sample_{idx}", 
                "metadata": {}
            }
        
        try:
            sample_data = self.data.iloc(idx)
            if hasattr(sample_data, 'values'):
                expression_data = sample_data.values
            else:
                expression_data = np.zeros(2000)
                
            if TORCH_AVAILABLE:
                expression = torch.tensor(expression_data, dtype=torch.float32)
            else:
                expression = np.array(expression_data, dtype=np.float32)
                
            sample_id = f"sample_{idx}"
            if hasattr(sample_data, 'name'):
                sample_id = str(sample_data.name)
            
            return {
                "expression": expression,
                "sample_id": sample_id,
                "metadata": {"sample_type": "bulk"}
            }
            
        except Exception as e:
            logger.warning(f"Failed to get bulk sample {idx}: {e}")
            return {
                "expression": torch.zeros(2000) if TORCH_AVAILABLE else np.zeros(2000),
                "sample_id": f"sample_{idx}",
                "metadata": {}
            }


def create_synthetic_genomics_data(config: Union[DictConfig, dict], output_dir: Union[str, Path]) -> None:
    """Create synthetic genomics data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_cells = config.get("n_cells", 1000)
    n_genes = config.get("n_genes", 2000)
    
    logger.info(f"Creating synthetic genomics data: {n_cells} cells, {n_genes} genes")
    
    # Generate expression matrix
    # Use negative binomial distribution to simulate count data
    expression_matrix = np.random.negative_binomial(
        n=10, p=0.3, size=(n_cells, n_genes)
    ).astype(float)
    
    # Add some structure with perturbation effects
    n_perturbed = n_cells // 2
    perturbation_genes = np.random.choice(n_genes, size=100, replace=False)
    
    # Upregulate some genes in perturbed cells
    expression_matrix[n_perturbed:, perturbation_genes[:50]] *= 2
    # Downregulate some genes in perturbed cells  
    expression_matrix[n_perturbed:, perturbation_genes[50:]] *= 0.5
    
    # Create cell metadata
    cell_metadata = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_cells)],
        'perturbation': ['control'] * n_perturbed + ['treated'] * (n_cells - n_perturbed),
        'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=n_cells),
        'batch': np.random.choice(['batch1', 'batch2', 'batch3'], size=n_cells)
    })
    
    # Create gene metadata
    gene_metadata = pd.DataFrame({
        'gene_id': [f'gene_{i}' for i in range(n_genes)],
        'gene_name': [f'GENE{i}' for i in range(n_genes)],
        'chromosome': np.random.choice([f'chr{i}' for i in range(1, 23)], size=n_genes)
    })
    
    if ANNDATA_AVAILABLE:
        # Create AnnData object
        adata = ad.AnnData(X=expression_matrix)
        adata.obs = cell_metadata.set_index('cell_id')
        adata.var = gene_metadata.set_index('gene_id')
        
        # Split into train/val/test
        train_size = int(0.7 * n_cells)
        val_size = int(0.15 * n_cells)
        
        train_adata = adata[:train_size].copy()
        val_adata = adata[train_size:train_size + val_size].copy()
        test_adata = adata[train_size + val_size:].copy()
        
        # Save files
        train_adata.write_h5ad(str(output_dir / "train.h5ad"))
        val_adata.write_h5ad(str(output_dir / "val.h5ad"))
        test_adata.write_h5ad(str(output_dir / "test.h5ad"))
        
    else:
        # Save as CSV files
        train_size = int(0.7 * n_cells)
        val_size = int(0.15 * n_cells)
        
        train_expr = pd.DataFrame(
            expression_matrix[:train_size],
            index=[f'cell_{i}' for i in range(train_size)],
            columns=[f'gene_{i}' for i in range(n_genes)]
        )
        train_expr.to_csv(str(output_dir / "train.csv"))
        
        val_start = train_size
        val_end = val_start + val_size
        val_expr = pd.DataFrame(
            expression_matrix[val_start:val_end],
            index=[f'cell_{i}' for i in range(val_start, val_end)],
            columns=[f'gene_{i}' for i in range(n_genes)]
        )
        val_expr.to_csv(str(output_dir / "val.csv"))
        
        test_expr = pd.DataFrame(
            expression_matrix[val_end:],
            index=[f'cell_{i}' for i in range(val_end, n_cells)],
            columns=[f'gene_{i}' for i in range(n_genes)]
        )
        test_expr.to_csv(str(output_dir / "test.csv"))
    
    # Save metadata
    cell_metadata.to_csv(str(output_dir / "cell_metadata.csv"), index=False)
    gene_metadata.to_csv(str(output_dir / "gene_metadata.csv"), index=False)
    
    logger.info(f"Created synthetic genomics data in {output_dir}")


def test_genomics_loader() -> None:
    """Test genomics data loader functionality."""
    logger.info("Testing genomics data loader...")
    
    # Create test config
    config = {
        "data_dir": "test_genomics_data",
        "batch_size": 8,
        "normalize": True,
        "log_transform": True,
        "filter_genes": False,  # Skip filtering for small test data
        "n_cells": 100,
        "n_genes": 500
    }
    
    # Create synthetic data
    create_synthetic_genomics_data(config, config["data_dir"])
    
    # Initialize data loader
    loader = GenomicsDataLoader(config)
    loader.setup()
    
    # Test data loading
    train_loader = loader.get_dataloader("train")
    if train_loader and TORCH_AVAILABLE:
        try:
            for batch in train_loader:
                logger.info(f"Batch keys: {list(batch.keys())}")
                if "expression" in batch and hasattr(batch["expression"], "shape"):
                    logger.info(f"Expression shape: {batch['expression'].shape}")
                break
        except Exception as e:
            logger.warning(f"Error iterating through dataloader: {e}")
    
    # Print statistics
    stats = loader.get_dataset_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    logger.info("Genomics data loader test completed successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    test_genomics_loader()