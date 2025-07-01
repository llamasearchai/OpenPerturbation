"""
PyTorch Lightning data modules for perturbation biology experiments.

Coordinates multi-modal data loading including imaging, genomics, and molecular data.
"""

import os
import torch
# Lazy import for pytorch lightning to avoid scipy issues
try:
    import pytorch_lightning as pl
    PYTORCH_LIGHTNING_AVAILABLE = True
except Exception:
    PYTORCH_LIGHTNING_AVAILABLE = False
    # Create dummy pl module
    class DummyLightningDataModule:
        def __init__(self):
            pass
        def setup(self, stage=None):
            pass
        def train_dataloader(self):
            return None
        def val_dataloader(self):
            return None
        def test_dataloader(self):
            return None
    
    class DummyPL:
        LightningDataModule = DummyLightningDataModule
    
    pl = DummyPL()
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict, List, Tuple, Optional, Union, Any, cast, Sized
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
import logging

# Import custom data loaders
from ..data.loaders.imaging_loader import HighContentImagingLoader
from ..data.loaders.genomics_loader import GenomicsDataLoader
from ..data.loaders.molecular_loader import MolecularDataLoader

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _safe_len(ds: Any) -> int:
    """Return len(ds) if possible, otherwise 0.

    This avoids static type checker issues when the Dataset implementation does
    not explicitly subclass Sized. At runtime, we still rely on `__len__` to be
    present – which is the case for all datasets used in the project.
    """

    try:
        # The cast ensures runtime safety; we suppress static complaint.
        return len(cast(Sized, ds))  # type: ignore[reportArgumentType]
    except Exception:
        return 0

class MultiModalDataset(Dataset[Dict[str, Any]]):
    """Dataset that combines imaging, genomics, and molecular data."""

    def __init__(
        self,
        imaging_dataset: Optional[Dataset[Any]] = None,
        genomics_dataset: Optional[Dataset[Any]] = None,
        molecular_dataset: Optional[Dataset[Any]] = None,
        config: Optional[DictConfig] = None,
    ):
        """
        Initialize multi-modal dataset.

        Args:
            imaging_dataset: High-content imaging dataset
            genomics_dataset: Genomics/transcriptomics dataset
            molecular_dataset: Molecular/chemical dataset
            config: Configuration dictionary
        """

        self.imaging_dataset = imaging_dataset
        self.genomics_dataset = genomics_dataset
        self.molecular_dataset = molecular_dataset
        self.config = config if config is not None else DictConfig({})

        # Determine dataset size (use largest available dataset)
        self.sizes = {}
        if imaging_dataset:
            self.sizes["imaging"] = _safe_len(imaging_dataset)
        if genomics_dataset:
            self.sizes["genomics"] = _safe_len(genomics_dataset)
        if molecular_dataset:
            self.sizes["molecular"] = _safe_len(molecular_dataset)

        if not self.sizes:
            raise ValueError("At least one dataset must be provided")

        self.size = max(self.sizes.values())
        self.primary_modality = max(self.sizes.keys(), key=lambda k: self.sizes[k])

        logger.info(f"Multi-modal dataset initialized with {self.size} samples")
        logger.info(f"Primary modality: {self.primary_modality}")
        logger.info(f"Available modalities: {list(self.sizes.keys())}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get multi-modal sample."""

        sample = {"sample_idx": idx, "modalities": list(self.sizes.keys())}

        # Get imaging data
        if self.imaging_dataset:
            imaging_idx = idx % _safe_len(self.imaging_dataset)
            imaging_sample = self.imaging_dataset[imaging_idx]
            sample["imaging"] = imaging_sample

        # Get genomics data
        if self.genomics_dataset:
            genomics_idx = idx % _safe_len(self.genomics_dataset)
            genomics_sample = self.genomics_dataset[genomics_idx]
            sample["genomics"] = genomics_sample

        # Get molecular data
        if self.molecular_dataset:
            molecular_idx = idx % _safe_len(self.molecular_dataset)
            molecular_sample = self.molecular_dataset[molecular_idx]
            sample["molecular"] = molecular_sample

        # Extract common perturbation information
        sample["perturbation"] = self._extract_perturbation_info(sample)

        return sample

    def _extract_perturbation_info(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and harmonize perturbation information across modalities."""

        perturbation_info = {
            "modalities_available": [],
            "perturbation_type": "unknown",
            "targets": [],
            "conditions": {},
        }

        # Check each modality for perturbation information
        for modality in ["imaging", "genomics", "molecular"]:
            if modality in sample:
                mod_sample = sample[modality]
                perturbation_info["modalities_available"].append(modality)

                if "perturbation" in mod_sample:
                    mod_pert = mod_sample["perturbation"]

                    # Extract perturbation type
                    if "type" in mod_pert:
                        perturbation_info["perturbation_type"] = mod_pert["type"]

                    # Extract targets
                    if "target" in mod_pert:
                        perturbation_info["targets"].append(mod_pert["target"])
                    elif "targets" in mod_pert:
                        perturbation_info["targets"].extend(mod_pert["targets"])

                    # Extract conditions
                    for key in ["concentration", "dose", "treatment_time", "cell_type"]:
                        if key in mod_pert:
                            perturbation_info["conditions"][key] = mod_pert[key]

        # Deduplicate targets
        perturbation_info["targets"] = list(set(perturbation_info["targets"]))

        return perturbation_info


class PerturbationDataModule(pl.LightningDataModule):
    """
    Lightning data module for perturbation biology experiments.

    Handles multi-modal data loading and preprocessing for imaging, genomics,
    and molecular data in perturbation biology studies.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Data paths
        self.data_dir = Path(config.get("data_dir", "data"))
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)

        # Dataset configurations
        self.imaging_config = config.get("imaging", {})
        self.genomics_config = config.get("genomics", {})
        self.molecular_config = config.get("molecular", {})

        # Data splits
        self.train_split = config.get("train_split", 0.7)
        self.val_split = config.get("val_split", 0.15)
        self.test_split = config.get("test_split", 0.15)

        # Initialize data loaders with proper type annotations
        self.imaging_loader: Optional[HighContentImagingLoader] = None
        self.genomics_loader: Optional[GenomicsDataLoader] = None
        self.molecular_loader: Optional[MolecularDataLoader] = None

        # Datasets with proper type annotations
        self.train_dataset: Optional[MultiModalDataset] = None
        self.val_dataset: Optional[MultiModalDataset] = None
        self.test_dataset: Optional[MultiModalDataset] = None

        # Cache for statistics
        self._data_statistics: Optional[Dict[str, Any]] = None

        logger.info(f"Initialized PerturbationDataModule with batch_size={self.batch_size}")

    def prepare_data(self):
        """Download and prepare data."""
        # Download datasets if necessary
        self._download_datasets()

        # Create synthetic data if needed
        if self.config.get("create_synthetic", False):
            self._create_synthetic_data()

        logger.info("Data preparation completed")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""

        logger.info(f"Setting up data for stage: {stage}")

        # Setup individual modality loaders
        self._setup_imaging_data()
        self._setup_genomics_data()
        self._setup_molecular_data()

        # Create multi-modal datasets
        if stage == "fit" or stage is None:
            self._create_train_val_datasets()

        if stage == "test" or stage is None:
            self._create_test_dataset()

        # Compute data statistics
        self._compute_data_statistics()

        logger.info("Data setup completed")

    def _setup_imaging_data(self):
        """Setup imaging data loader."""
        if self.imaging_config:
            try:
                self.imaging_loader = HighContentImagingLoader(self.imaging_config)
                self.imaging_loader.setup()
                logger.info("Imaging data loader initialized")
            except Exception as e:
                logger.warning(f"Could not initialize imaging loader: {e}")
                self.imaging_loader = None

    def _setup_genomics_data(self):
        """Setup genomics data loader."""
        if self.genomics_config:
            try:
                self.genomics_loader = GenomicsDataLoader(self.genomics_config)
                self.genomics_loader.setup()
                logger.info("Genomics data loader initialized")
            except Exception as e:
                logger.warning(f"Could not initialize genomics loader: {e}")
                self.genomics_loader = None

    def _setup_molecular_data(self):
        """Setup molecular data loader."""
        if self.molecular_config:
            try:
                self.molecular_loader = MolecularDataLoader(self.molecular_config)
                self.molecular_loader.setup()
                logger.info("Molecular data loader initialized")
            except Exception as e:
                logger.warning(f"Could not initialize molecular loader: {e}")
                self.molecular_loader = None

    def _create_train_val_datasets(self):
        """Create training and validation datasets."""

        # Get individual datasets using getattr with fallbacks
        imaging_dataset = getattr(self.imaging_loader, "train_dataset", None) if self.imaging_loader else None
        genomics_dataset = getattr(self.genomics_loader, "train_dataset", None) if self.genomics_loader else None
        molecular_dataset = getattr(self.molecular_loader, "train_dataset", None) if self.molecular_loader else None

        # Create combined training dataset
        self.train_dataset = MultiModalDataset(
            imaging_dataset=cast(Optional[Dataset[Any]], imaging_dataset),
            genomics_dataset=cast(Optional[Dataset[Any]], genomics_dataset),
            molecular_dataset=cast(Optional[Dataset[Any]], molecular_dataset),
            config=self.config,
        )

        # Create validation dataset
        imaging_val = getattr(self.imaging_loader, "val_dataset", None) if self.imaging_loader else None
        genomics_val = getattr(self.genomics_loader, "val_dataset", None) if self.genomics_loader else None
        molecular_val = getattr(self.molecular_loader, "val_dataset", None) if self.molecular_loader else None

        self.val_dataset = MultiModalDataset(
            imaging_dataset=cast(Optional[Dataset[Any]], imaging_val),
            genomics_dataset=cast(Optional[Dataset[Any]], genomics_val),
            molecular_dataset=cast(Optional[Dataset[Any]], molecular_val),
            config=self.config,
        )

        logger.info(f"Created training dataset with {len(self.train_dataset)} samples")
        logger.info(f"Created validation dataset with {len(self.val_dataset)} samples")

    def _create_test_dataset(self):
        """Create test dataset."""

        # Get individual test datasets using getattr with fallbacks
        imaging_test = getattr(self.imaging_loader, "test_dataset", None) if self.imaging_loader else None
        genomics_test = getattr(self.genomics_loader, "test_dataset", None) if self.genomics_loader else None
        molecular_test = getattr(self.molecular_loader, "test_dataset", None) if self.molecular_loader else None

        self.test_dataset = MultiModalDataset(
            imaging_dataset=cast(Optional[Dataset[Any]], imaging_test),
            genomics_dataset=cast(Optional[Dataset[Any]], genomics_test),
            molecular_dataset=cast(Optional[Dataset[Any]], molecular_test),
            config=self.config,
        )

        logger.info(f"Created test dataset with {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for multi-modal data."""

        # Initialize batch dictionary
        collated_batch: Dict[str, Any] = {"batch_size": len(batch), "modalities": set(), "sample_indices": []}

        # Collect sample indices
        for sample in batch:
            collated_batch["sample_indices"].append(sample["sample_idx"])
            collated_batch["modalities"].update(sample["modalities"])

        collated_batch["modalities"] = list(collated_batch["modalities"])

        # Collate each modality
        for modality in collated_batch["modalities"]:
            modality_samples = [sample[modality] for sample in batch if modality in sample]

            if modality_samples:
                collated_batch[modality] = self._collate_modality_data(modality_samples, modality)

        # Collate perturbation information
        perturbation_samples = [sample["perturbation"] for sample in batch]
        collated_batch["perturbation"] = self._collate_perturbation_data(perturbation_samples)

        return collated_batch

    def _collate_modality_data(self, samples: List[Dict[str, Any]], modality: str) -> Dict[str, Any]:
        """Collate data for a specific modality."""

        if modality == "imaging":
            return self._collate_imaging_data(samples)
        elif modality == "genomics":
            return self._collate_genomics_data(samples)
        elif modality == "molecular":
            return self._collate_molecular_data(samples)
        else:
            logger.warning(f"Unknown modality: {modality}")
            return {}

    def _collate_imaging_data(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate imaging data."""

        # Stack images
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        metadata: List[Dict[str, Any]] = []

        for sample in samples:
            if "image" in sample:
                images.append(sample["image"])
            if "mask" in sample:
                masks.append(sample["mask"])
            if "metadata" in sample:
                metadata.append(sample["metadata"])

        collated: Dict[str, Any] = {}

        if images:
            collated["images"] = torch.stack(images)

        if masks:
            collated["masks"] = torch.stack(masks)

        if metadata:
            collated["metadata"] = metadata

        return collated

    def _collate_genomics_data(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate genomics data."""

        # Stack expression data
        expressions: List[torch.Tensor] = []
        gene_names: List[List[str]] = []
        metadata: List[Dict[str, Any]] = []

        for sample in samples:
            if "expression" in sample:
                expressions.append(sample["expression"])
            if "gene_names" in sample:
                gene_names.append(sample["gene_names"])
            if "metadata" in sample:
                metadata.append(sample["metadata"])

        collated: Dict[str, Any] = {}

        if expressions:
            collated["expressions"] = torch.stack(expressions)

        if gene_names:
            collated["gene_names"] = gene_names

        if metadata:
            collated["metadata"] = metadata

        return collated

    def _collate_molecular_data(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate molecular data."""

        # Collect molecular features
        smiles: List[str] = []
        descriptors: List[Dict[str, float]] = []
        fingerprints: List[Any] = []
        graphs: List[Any] = []

        for sample in samples:
            if "smiles" in sample:
                smiles.append(sample["smiles"])

            if "molecular_features" in sample:
                mol_features = sample["molecular_features"]

                if "descriptors" in mol_features:
                    descriptors.append(mol_features["descriptors"])

                if "fingerprints" in mol_features:
                    fingerprints.append(mol_features["fingerprints"])

                if "graph" in mol_features and mol_features["graph"] is not None:
                    graphs.append(mol_features["graph"])

        collated: Dict[str, Any] = {}

        if smiles:
            collated["smiles"] = smiles

        if descriptors:
            # Convert descriptor dicts to arrays
            descriptor_arrays = []
            for desc_dict in descriptors:
                desc_array = [
                    desc_dict.get(name, 0.0)
                    for name in self.molecular_config.get("features", {}).get("descriptors", [])
                ]
                descriptor_arrays.append(desc_array)
            collated["descriptors"] = torch.tensor(descriptor_arrays, dtype=torch.float)

        if fingerprints:
            collated["fingerprints"] = torch.tensor(fingerprints, dtype=torch.float)

        if graphs:
            try:
                from torch_geometric.data import Batch

                collated["graphs"] = Batch.from_data_list(graphs)
            except ImportError:
                logger.warning("PyTorch Geometric not available for graph batching")

        return collated

    def _collate_perturbation_data(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate perturbation information."""

        # Collect all perturbation information
        all_modalities = set()
        all_types = set()
        all_targets = set()
        all_conditions = {}

        for sample in samples:
            all_modalities.update(sample.get("modalities_available", []))
            all_types.add(sample.get("perturbation_type", "unknown"))
            all_targets.update(sample.get("targets", []))

            # Collect conditions
            conditions = sample.get("conditions", {})
            for key, value in conditions.items():
                if key not in all_conditions:
                    all_conditions[key] = []
                all_conditions[key].append(value)

        return {
            "modalities_available": list(all_modalities),
            "perturbation_types": list(all_types),
            "targets": list(all_targets),
            "conditions": all_conditions,
            "batch_size": len(samples),
        }

    def _compute_data_statistics(self):
        """Compute and cache data statistics."""

        stats: Dict[str, Any] = {
            "total_samples": {
                "train": len(self.train_dataset) if self.train_dataset else 0,
                "val": len(self.val_dataset) if self.val_dataset else 0,
                "test": len(self.test_dataset) if self.test_dataset else 0,
            },
            "modalities": {
                "imaging": self.imaging_loader is not None,
                "genomics": self.genomics_loader is not None,
                "molecular": self.molecular_loader is not None,
            },
            "config": dict(self.config),
        }

        # Get modality-specific statistics
        if self.imaging_loader:
            stats["imaging_stats"] = self.imaging_loader.get_dataset_statistics()

        if self.genomics_loader:
            try:
                stats["genomics_stats"] = self.genomics_loader.get_dataset_statistics()
            except AttributeError:
                stats["genomics_stats"] = {"error": "Statistics not available"}

        if self.molecular_loader:
            try:
                stats["molecular_stats"] = self.molecular_loader.get_dataset_statistics()
            except AttributeError:
                stats["molecular_stats"] = {"error": "Statistics not available"}

        # Check data balance
        train_size = stats["total_samples"]["train"]
        val_size = stats["total_samples"]["val"]
        test_size = stats["total_samples"]["test"]

        if train_size > 0 and val_size > 0 and test_size > 0:
            total = train_size + val_size + test_size
            stats["data_splits"] = {
                "train": train_size / total,
                "val": val_size / total,
                "test": test_size / total,
            }

            splits_dict = {
                "train": train_size,
                "val": val_size,
                "test": test_size,
            }
            stats["balanced_splits"] = self._check_balanced_splits(splits_dict)

        self._data_statistics = stats
        logger.info("Data statistics computed")

    def _check_balanced_splits(self, sizes: Dict[str, int]) -> bool:
        """Check if data splits are reasonably balanced."""

        total = sum(sizes.values())
        if total == 0:
            return False

        # Check if splits are within reasonable ranges
        train_ratio = sizes["train"] / total
        val_ratio = sizes["val"] / total
        test_ratio = sizes["test"] / total

        # Reasonable ranges
        train_ok = 0.6 <= train_ratio <= 0.8
        val_ok = 0.1 <= val_ratio <= 0.25
        test_ok = 0.1 <= test_ratio <= 0.25

        return train_ok and val_ok and test_ok

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics."""
        if self._data_statistics is None:
            self._compute_data_statistics()
        return self._data_statistics or {}

    def _create_synthetic_data(self):
        """Create synthetic data for testing."""

        logger.info("Creating synthetic data...")

        # Create synthetic imaging data
        if self.imaging_config.get("create_synthetic", False):
            from ..data.loaders.imaging_loader import create_synthetic_imaging_data

            imaging_dir = self.data_dir / "imaging"
            imaging_dir.mkdir(parents=True, exist_ok=True)

            create_synthetic_imaging_data(config=self.imaging_config, output_dir=str(imaging_dir))

        # Create synthetic genomics data
        if self.genomics_config.get("create_synthetic", False):
            from ..data.loaders.genomics_loader import create_synthetic_genomics_data

            genomics_dir = self.data_dir / "genomics"
            genomics_dir.mkdir(parents=True, exist_ok=True)

            create_synthetic_genomics_data(
                config=self.genomics_config, output_dir=str(genomics_dir)
            )

        # Create synthetic molecular data
        if self.molecular_config.get("create_synthetic", False):
            from ..data.loaders.molecular_loader import create_synthetic_molecular_data

            molecular_dir = self.data_dir / "molecular"
            molecular_dir.mkdir(parents=True, exist_ok=True)

            create_synthetic_molecular_data(
                config=self.molecular_config, output_dir=str(molecular_dir)
            )

    def _download_datasets(self):
        """Download datasets from URLs."""

        download_urls = self.config.get("download_urls", {})

        if not download_urls:
            return

        logger.info("Downloading datasets...")

        import requests
        from tqdm import tqdm

        for dataset_name, url in download_urls.items():
            output_path = self.data_dir / f"{dataset_name}.zip"

            if output_path.exists():
                logger.info(f"Dataset {dataset_name} already exists, skipping download")
                continue

            logger.info(f"Downloading {dataset_name} from {url}")

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(output_path, "wb") as f, tqdm(
                    desc=dataset_name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)

                # Extract if it's a zip file
                if output_path.suffix == ".zip":
                    import zipfile

                    extract_dir = self.data_dir / dataset_name
                    extract_dir.mkdir(exist_ok=True)

                    with zipfile.ZipFile(output_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)

                    logger.info(f"Extracted {dataset_name} to {extract_dir}")

            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")

    def get_sample_batch(self, split: str = "train", batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Get a sample batch for inspection."""

        if batch_size is None:
            batch_size = min(4, self.batch_size)

        if split == "train" and self.train_dataset:
            dataset = self.train_dataset
        elif split == "val" and self.val_dataset:
            dataset = self.val_dataset
        elif split == "test" and self.test_dataset:
            dataset = self.test_dataset
        else:
            logger.warning(f"Dataset for split '{split}' not available")
            return {}

        # Sample random indices
        indices = np.random.choice(len(dataset), min(batch_size, len(dataset)), replace=False)
        samples = [dataset[idx] for idx in indices]

        # Collate samples
        batch = self._collate_fn(samples)

        return batch

    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate data integrity across modalities."""

        integrity_checks = {
            "train_dataset_exists": self.train_dataset is not None,
            "val_dataset_exists": self.val_dataset is not None,
            "datasets_non_empty": True,
            "modalities_consistent": True,
            "perturbation_info_available": True,
            "batch_collation_works": True,
        }

        # Check if datasets are non-empty
        if self.train_dataset and len(self.train_dataset) == 0:
            integrity_checks["datasets_non_empty"] = False
        if self.val_dataset and len(self.val_dataset) == 0:
            integrity_checks["datasets_non_empty"] = False

        # Test batch collation
        try:
            sample_batch = self.get_sample_batch("train", batch_size=2)
            if not sample_batch:
                integrity_checks["batch_collation_works"] = False
        except Exception as e:
            logger.warning(f"Batch collation test failed: {e}")
            integrity_checks["batch_collation_works"] = False

        # Check modality consistency
        try:
            if self.train_dataset:
                sample = self.train_dataset[0]
                if "perturbation" not in sample:
                    integrity_checks["perturbation_info_available"] = False

                available_modalities = sample.get("modalities", [])
                if len(available_modalities) == 0:
                    integrity_checks["modalities_consistent"] = False
        except Exception as e:
            logger.warning(f"Modality consistency check failed: {e}")
            integrity_checks["modalities_consistent"] = False

        return integrity_checks

    def get_modality_sample(self, modality: str, split: str = "train") -> Optional[Dict[str, Any]]:
        """Get a sample from a specific modality."""

        if split == "train" and self.train_dataset:
            dataset = self.train_dataset
        elif split == "val" and self.val_dataset:
            dataset = self.val_dataset
        elif split == "test" and self.test_dataset:
            dataset = self.test_dataset
        else:
            return None

        # Find a sample that contains the requested modality
        for i in range(min(100, len(dataset))):  # Check first 100 samples
            sample = dataset[i]
            if modality in sample.get("modalities", []):
                return sample[modality]

        logger.warning(f"No sample found containing modality: {modality}")
        return None

    def export_data_summary(self, output_path: str):
        """Export data summary to file."""

        stats = self.get_data_statistics()
        integrity = self.validate_data_integrity()

        summary = {
            "experiment_config": dict(self.config),
            "data_statistics": stats,
            "integrity_checks": integrity,
            "export_timestamp": pd.Timestamp.now().isoformat(),
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if output_path_obj.suffix == ".json":
            import json

            with open(output_path_obj, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        else:
            # Export as YAML
            import yaml

            with open(output_path_obj, "w") as f:
                yaml.dump(summary, f, default_flow_style=False)

        logger.info(f"Data summary exported to: {output_path_obj}")


class SingleModalityDataModule(pl.LightningDataModule):
    """Data module for single modality experiments."""

    def __init__(self, config: DictConfig, modality: str):
        super().__init__()
        self.config = config
        self.modality = modality
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)

        # Initialize appropriate loader
        if modality == "imaging":
            self.data_loader = HighContentImagingLoader(config)
        elif modality == "genomics":
            self.data_loader = GenomicsDataLoader(config)
        elif modality == "molecular":
            self.data_loader = MolecularDataLoader(config)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def setup(self, stage: Optional[str] = None):
        """Setup single modality data."""
        self.data_loader.setup(stage)

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return training dataloader."""
        loader = getattr(self.data_loader, "train_dataloader", None)
        if loader is None:
            raise RuntimeError(f"Training dataloader not available for {self.modality}")
        if callable(loader):
            result = loader()
        else:
            result = loader
        
        if result is None:
            raise RuntimeError(f"Training dataloader returned None for {self.modality}")
        return cast(DataLoader[Dict[str, Any]], result)

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return validation dataloader."""
        loader = getattr(self.data_loader, "val_dataloader", None)
        if loader is None:
            raise RuntimeError(f"Validation dataloader not available for {self.modality}")
        if callable(loader):
            result = loader()
        else:
            result = loader
        
        if result is None:
            raise RuntimeError(f"Validation dataloader returned None for {self.modality}")
        return cast(DataLoader[Dict[str, Any]], result)

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return test dataloader."""
        loader = getattr(self.data_loader, "test_dataloader", None)
        if loader is None:
            raise RuntimeError(f"Test dataloader not available for {self.modality}")
        if callable(loader):
            result = loader()
        else:
            result = loader
        
        if result is None:
            raise RuntimeError(f"Test dataloader returned None for {self.modality}")
        return cast(DataLoader[Dict[str, Any]], result)


class HighContentImagingDataset(Dataset[Dict[str, Any]]):
    """High-content imaging dataset class."""
    
    def __init__(self, config: DictConfig, metadata_file: str, data_dir: str, mode: str = "train"):
        """Initialize the imaging dataset."""
        self.config = config
        self.metadata_file = metadata_file
        self.data_dir = data_dir
        self.mode = mode
        
        # Load metadata
        try:
            self.metadata = pd.read_csv(metadata_file)
        except Exception:
            # Create dummy metadata
            self.metadata = pd.DataFrame({
                'sample_id': [f'sample_{i}' for i in range(100)],
                'split': [mode] * 100
            })
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        row = self.metadata.iloc[idx]
        return {
            'sample_id': row.get('sample_id', f'sample_{idx}'),
            'metadata': row.to_dict(),
            'image': torch.zeros(3, 224, 224),  # Dummy image
            'mask': None
        }


class BaseDataModule(pl.LightningDataModule):
    """Base data module with common functionality."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataset: Optional[Any] = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup the dataset - to be implemented by subclasses."""
        pass
        
    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return training dataloader."""
        if not self.dataset:
            self.setup()
        if self.dataset is None or not hasattr(self.dataset, 'train'):
            raise RuntimeError("Dataset not properly initialized")
            
        return DataLoader(
            self.dataset.train, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return validation dataloader."""
        if not self.dataset:
            self.setup()
        if self.dataset is None or not hasattr(self.dataset, 'val'):
            raise RuntimeError("Dataset not properly initialized")
            
        return DataLoader(
            self.dataset.val, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
    
    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return test dataloader."""
        if not self.dataset:
            self.setup()
        if self.dataset is None or not hasattr(self.dataset, 'test'):
            raise RuntimeError("Dataset not properly initialized")
            
        return DataLoader(
            self.dataset.test, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
