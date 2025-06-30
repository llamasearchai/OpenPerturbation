"""
PyTorch Lightning data modules for perturbation biology experiments.

Coordinates multi-modal data loading including imaging, genomics, and molecular data.
"""

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict, List, Tuple, Optional, Union, Any
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


class MultiModalDataset(Dataset):
    """Dataset that combines imaging, genomics, and molecular data."""

    def __init__(
        self,
        imaging_dataset: Optional[Dataset] = None,
        genomics_dataset: Optional[Dataset] = None,
        molecular_dataset: Optional[Dataset] = None,
        config: DictConfig = None,
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
        self.config = config or DictConfig({})

        # Determine dataset size (use largest available dataset)
        self.sizes = {}
        if imaging_dataset:
            self.sizes["imaging"] = len(imaging_dataset)
        if genomics_dataset:
            self.sizes["genomics"] = len(genomics_dataset)
        if molecular_dataset:
            self.sizes["molecular"] = len(molecular_dataset)

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
            imaging_idx = idx % len(self.imaging_dataset)
            imaging_sample = self.imaging_dataset[imaging_idx]
            sample["imaging"] = imaging_sample

        # Get genomics data
        if self.genomics_dataset:
            genomics_idx = idx % len(self.genomics_dataset)
            genomics_sample = self.genomics_dataset[genomics_idx]
            sample["genomics"] = genomics_sample

        # Get molecular data
        if self.molecular_dataset:
            molecular_idx = idx % len(self.molecular_dataset)
            molecular_sample = self.molecular_dataset[molecular_idx]
            sample["molecular"] = molecular_sample

        # Extract common perturbation information
        sample["perturbation"] = self._extract_perturbation_info(sample)

        return sample

    def _extract_perturbation_info(self, sample: Dict) -> Dict:
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

        # Initialize data loaders
        self.imaging_loader = None
        self.genomics_loader = None
        self.molecular_loader = None

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Cache for statistics
        self._data_statistics = None

        logger.info(f"Initialized PerturbationDataModule with batch_size={self.batch_size}")

    def prepare_data(self):
        """Download and prepare data if needed."""

        # Create data directories
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check if synthetic data should be created
        if self.config.get("create_synthetic_data", False):
            self._create_synthetic_data()

        # Download datasets if URLs are provided
        if self.config.get("download_urls"):
            self._download_datasets()

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, or testing."""

        logger.info(f"Setting up data module for stage: {stage}")

        # Initialize individual data loaders
        self._setup_imaging_data()
        self._setup_genomics_data()
        self._setup_molecular_data()

        # Create multi-modal datasets
        if stage == "fit" or stage is None:
            self._create_train_val_datasets()

        if stage == "test" or stage is None:
            self._create_test_dataset()

        # Compute and cache statistics
        self._compute_data_statistics()

        logger.info("Data module setup completed")

    def _setup_imaging_data(self):
        """Setup high-content imaging data."""

        if self.imaging_config.get("enabled", True):
            try:
                logger.info("Setting up imaging data...")

                # Configure imaging loader
                imaging_config = DictConfig(
                    {
                        **self.imaging_config,
                        "data_dir": str(self.data_dir / "imaging"),
                        "batch_size": self.batch_size,
                        "num_workers": self.num_workers,
                        "pin_memory": self.pin_memory,
                    }
                )

                self.imaging_loader = HighContentImagingLoader(imaging_config)
                self.imaging_loader.setup()

                logger.info(f"Imaging data loaded: {self.imaging_loader.get_dataset_statistics()}")

            except Exception as e:
                logger.warning(f"Failed to setup imaging data: {e}")
                self.imaging_loader = None

    def _setup_genomics_data(self):
        """Setup genomics/transcriptomics data."""

        if self.genomics_config.get("enabled", True):
            try:
                logger.info("Setting up genomics data...")

                # Configure genomics loader
                genomics_config = DictConfig(
                    {
                        **self.genomics_config,
                        "data_dir": str(self.data_dir / "genomics"),
                        "batch_size": self.batch_size,
                        "num_workers": self.num_workers,
                        "pin_memory": self.pin_memory,
                    }
                )

                self.genomics_loader = GenomicsDataLoader(genomics_config)
                self.genomics_loader.setup()

                logger.info(
                    f"Genomics data loaded: {self.genomics_loader.get_dataset_statistics()}"
                )

            except Exception as e:
                logger.warning(f"Failed to setup genomics data: {e}")
                self.genomics_loader = None

    def _setup_molecular_data(self):
        """Setup molecular/chemical data."""

        if self.molecular_config.get("enabled", True):
            try:
                logger.info("Setting up molecular data...")

                # Configure molecular loader
                molecular_config = DictConfig(
                    {
                        **self.molecular_config,
                        "data_file": str(self.data_dir / "molecular" / "compounds.csv"),
                        "metadata_file": str(self.data_dir / "molecular" / "metadata.csv"),
                        "batch_size": self.batch_size,
                        "num_workers": self.num_workers,
                        "pin_memory": self.pin_memory,
                    }
                )

                self.molecular_loader = MolecularDataLoader(molecular_config)
                self.molecular_loader.setup()

                logger.info(
                    f"Molecular data loaded: {self.molecular_loader.get_dataset_statistics()}"
                )

            except Exception as e:
                logger.warning(f"Failed to setup molecular data: {e}")
                self.molecular_loader = None

    def _create_train_val_datasets(self):
        """Create training and validation datasets."""

        # Collect datasets from each modality
        train_datasets = {}
        val_datasets = {}

        if self.imaging_loader:
            train_datasets["imaging"] = self.imaging_loader.get_dataset("train")
            val_datasets["imaging"] = self.imaging_loader.get_dataset("val")

        if self.genomics_loader:
            train_datasets["genomics"] = self.genomics_loader.get_dataset("train")
            val_datasets["genomics"] = self.genomics_loader.get_dataset("val")

        if self.molecular_loader:
            train_datasets["molecular"] = self.molecular_loader.datasets["train"]
            val_datasets["molecular"] = self.molecular_loader.datasets["val"]

        # Create multi-modal datasets
        self.train_dataset = MultiModalDataset(
            imaging_dataset=train_datasets.get("imaging"),
            genomics_dataset=train_datasets.get("genomics"),
            molecular_dataset=train_datasets.get("molecular"),
            config=self.config,
        )

        self.val_dataset = MultiModalDataset(
            imaging_dataset=val_datasets.get("imaging"),
            genomics_dataset=val_datasets.get("genomics"),
            molecular_dataset=val_datasets.get("molecular"),
            config=self.config,
        )

        logger.info(f"Created training dataset with {len(self.train_dataset)} samples")
        logger.info(f"Created validation dataset with {len(self.val_dataset)} samples")

    def _create_test_dataset(self):
        """Create test dataset."""

        # Collect test datasets from each modality
        test_datasets = {}

        if self.imaging_loader:
            test_datasets["imaging"] = self.imaging_loader.get_dataset("test")

        if self.genomics_loader:
            test_datasets["genomics"] = self.genomics_loader.get_dataset("test")

        if self.molecular_loader:
            test_datasets["molecular"] = self.molecular_loader.datasets["test"]

        # Create multi-modal test dataset
        self.test_dataset = MultiModalDataset(
            imaging_dataset=test_datasets.get("imaging"),
            genomics_dataset=test_datasets.get("genomics"),
            molecular_dataset=test_datasets.get("molecular"),
            config=self.config,
        )

        logger.info(f"Created test dataset with {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for multi-modal data."""

        # Initialize batch dictionary
        collated_batch = {"batch_size": len(batch), "modalities": set(), "sample_indices": []}

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

    def _collate_modality_data(self, samples: List[Dict], modality: str) -> Dict[str, Any]:
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

    def _collate_imaging_data(self, samples: List[Dict]) -> Dict[str, Any]:
        """Collate imaging data."""

        # Stack images
        images = []
        masks = []
        metadata = []

        for sample in samples:
            if "image" in sample:
                images.append(sample["image"])
            if "mask" in sample:
                masks.append(sample["mask"])
            if "metadata" in sample:
                metadata.append(sample["metadata"])

        collated = {}

        if images:
            collated["images"] = torch.stack(images)

        if masks:
            collated["masks"] = torch.stack(masks)

        if metadata:
            collated["metadata"] = metadata

        return collated

    def _collate_genomics_data(self, samples: List[Dict]) -> Dict[str, Any]:
        """Collate genomics data."""

        # Stack expression data
        expressions = []
        gene_names = []
        metadata = []

        for sample in samples:
            if "expression" in sample:
                expressions.append(sample["expression"])
            if "gene_names" in sample:
                gene_names.append(sample["gene_names"])
            if "metadata" in sample:
                metadata.append(sample["metadata"])

        collated = {}

        if expressions:
            collated["expressions"] = torch.stack(expressions)

        if gene_names:
            collated["gene_names"] = gene_names

        if metadata:
            collated["metadata"] = metadata

        return collated

    def _collate_molecular_data(self, samples: List[Dict]) -> Dict[str, Any]:
        """Collate molecular data."""

        # Collect molecular features
        smiles = []
        descriptors = []
        fingerprints = []
        graphs = []

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

        collated = {}

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

    def _collate_perturbation_data(self, samples: List[Dict]) -> Dict[str, Any]:
        """Collate perturbation information."""

        # Collect perturbation data
        perturbation_types = []
        all_targets = []
        modalities_available = []
        conditions = []

        for sample in samples:
            perturbation_types.append(sample.get("perturbation_type", "unknown"))
            all_targets.extend(sample.get("targets", []))
            modalities_available.append(sample.get("modalities_available", []))
            conditions.append(sample.get("conditions", {}))

        return {
            "perturbation_types": perturbation_types,
            "all_targets": list(set(all_targets)),  # Unique targets
            "modalities_available": modalities_available,
            "conditions": conditions,
            "num_unique_perturbation_types": len(set(perturbation_types)),
            "num_unique_targets": len(set(all_targets)),
        }

    def _compute_data_statistics(self):
        """Compute and cache dataset statistics."""

        logger.info("Computing dataset statistics...")

        stats = {
            "dataset_sizes": {},
            "modality_coverage": {},
            "perturbation_statistics": {},
            "data_quality": {},
        }

        # Dataset sizes
        if self.train_dataset:
            stats["dataset_sizes"]["train"] = len(self.train_dataset)
        if self.val_dataset:
            stats["dataset_sizes"]["val"] = len(self.val_dataset)
        if self.test_dataset:
            stats["dataset_sizes"]["test"] = len(self.test_dataset)

        # Modality coverage
        available_modalities = []
        if self.imaging_loader:
            available_modalities.append("imaging")
        if self.genomics_loader:
            available_modalities.append("genomics")
        if self.molecular_loader:
            available_modalities.append("molecular")

        stats["modality_coverage"]["available_modalities"] = available_modalities
        stats["modality_coverage"]["num_modalities"] = len(available_modalities)

        # Individual modality statistics
        if self.imaging_loader:
            stats["modality_coverage"]["imaging"] = self.imaging_loader.get_dataset_statistics()
        if self.genomics_loader:
            stats["modality_coverage"]["genomics"] = self.genomics_loader.get_dataset_statistics()
        if self.molecular_loader:
            stats["modality_coverage"]["molecular"] = self.molecular_loader.get_dataset_statistics()

        # Sample some data to get perturbation statistics
        if self.train_dataset and len(self.train_dataset) > 0:
            sample_size = min(100, len(self.train_dataset))
            sample_indices = np.random.choice(len(self.train_dataset), sample_size, replace=False)

            perturbation_types = []
            all_targets = []
            modalities_per_sample = []

            for idx in sample_indices:
                sample = self.train_dataset[idx]
                pert_info = sample["perturbation"]

                perturbation_types.append(pert_info.get("perturbation_type", "unknown"))
                all_targets.extend(pert_info.get("targets", []))
                modalities_per_sample.append(len(pert_info.get("modalities_available", [])))

            stats["perturbation_statistics"] = {
                "unique_perturbation_types": list(set(perturbation_types)),
                "num_perturbation_types": len(set(perturbation_types)),
                "unique_targets": list(set(all_targets)),
                "num_targets": len(set(all_targets)),
                "avg_modalities_per_sample": np.mean(modalities_per_sample),
                "perturbation_type_distribution": {
                    ptype: perturbation_types.count(ptype) for ptype in set(perturbation_types)
                },
            }

        # Data quality checks
        stats["data_quality"] = {
            "all_modalities_available": len(available_modalities) >= 2,
            "sufficient_train_data": stats["dataset_sizes"].get("train", 0) >= 100,
            "balanced_splits": self._check_balanced_splits(stats["dataset_sizes"]),
        }

        self._data_statistics = stats

        logger.info(f"Dataset statistics computed:")
        logger.info(f"  Available modalities: {available_modalities}")
        logger.info(f"  Dataset sizes: {stats['dataset_sizes']}")
        logger.info(f"  Data quality: {stats['data_quality']}")

    def _check_balanced_splits(self, sizes: Dict[str, int]) -> bool:
        """Check if dataset splits are reasonably balanced."""

        if len(sizes) < 2:
            return False

        total_size = sum(sizes.values())
        if total_size == 0:
            return False

        # Check if splits are within reasonable ranges
        train_ratio = sizes.get("train", 0) / total_size
        val_ratio = sizes.get("val", 0) / total_size
        test_ratio = sizes.get("test", 0) / total_size

        # Reasonable ranges for splits
        return 0.5 <= train_ratio <= 0.8 and 0.1 <= val_ratio <= 0.3 and 0.1 <= test_ratio <= 0.3

    def get_data_statistics(self) -> Dict:
        """Get cached dataset statistics."""

        if self._data_statistics is None:
            self._compute_data_statistics()

        return self._data_statistics

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

    def get_sample_batch(self, split: str = "train", batch_size: Optional[int] = None) -> Dict:
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

    def get_modality_sample(self, modality: str, split: str = "train") -> Optional[Dict]:
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

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            import json

            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        else:
            # Export as YAML
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(summary, f, default_flow_style=False)

        logger.info(f"Data summary exported to: {output_path}")


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

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.modality == "molecular":
            return self.data_loader.get_dataloader("train")
        else:
            return self.data_loader.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.modality == "molecular":
            return self.data_loader.get_dataloader("val")
        else:
            return self.data_loader.get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.modality == "molecular":
            return self.data_loader.get_dataloader("test")
        else:
            return self.data_loader.get_dataloader("test")


class HighContentImagingDataset(Dataset):
    def __len__(self) -> int:
        return len(self.metadata)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataset = None
        
    def setup(self, stage: Optional[str] = None):
        # Implementation specific to each dataset
        pass
        
    def train_dataloader(self) -> DataLoader:
        if not self.dataset:
            self.setup()
        return DataLoader(
            self.dataset.train, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
    # Similar implementations for val_dataloader and test_dataloader
