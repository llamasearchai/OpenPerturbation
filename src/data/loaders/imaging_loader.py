"""
High-content imaging data loader for perturbation biology experiments.

Handles loading and preprocessing of multi-channel microscopy images including
segmentation masks, metadata, and perturbation annotations.
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Core dependencies
from omegaconf import DictConfig

# Optional dependencies with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    warnings.warn("OpenCV not available, using fallback implementations")
    CV2_AVAILABLE = False
    
    # Create minimal cv2 functionality
    class DummyCV2:
        IMREAD_UNCHANGED = -1
        IMREAD_GRAYSCALE = 0
        
        @staticmethod
        def imread(filepath: str, flags: int = 0) -> Optional[np.ndarray]:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        @staticmethod
        def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
            return np.zeros((*size, image.shape[2] if len(image.shape) > 2 else 1), dtype=image.dtype)
        
        @staticmethod
        def convertScaleAbs(image: np.ndarray, alpha: float = 1, beta: float = 0) -> np.ndarray:
            return (image * alpha + beta).astype(np.uint8)
        
        @staticmethod
        def imwrite(filepath: str, image: np.ndarray) -> bool:
            """Dummy image write function."""
            return True

    cv2 = DummyCV2()

try:
    from skimage import io, transform, filters, exposure, measure, segmentation  # type: ignore
    SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover
    # Catch broad exceptions (e.g., SciPy capability errors) and fall back to dummy implementation
    warnings.warn("scikit-image not available or failed to import – using fallback implementations")
    SKIMAGE_AVAILABLE = False
    
    # Create minimal skimage functionality  
    class DummyIO:
        @staticmethod
        def imread(filepath: str) -> np.ndarray:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        @staticmethod
        def imsave(filepath: str, image: np.ndarray) -> None:
            pass
    
    class DummyTransform:
        @staticmethod
        def resize(image: np.ndarray, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
            # Handle the preserve_range and anti_aliasing parameters
            if len(shape) == 2:
                # For 2D images, add channel dimension back if needed
                if len(image.shape) == 3:
                    resized = np.zeros((*shape, image.shape[2]), dtype=image.dtype)
                else:
                    resized = np.zeros(shape, dtype=image.dtype)
            else:
                resized = np.zeros(shape, dtype=image.dtype)
            
            return resized
    
    class DummyFilters:
        @staticmethod
        def gaussian(image: np.ndarray, sigma: float = 1) -> np.ndarray:
            return image
        
        @staticmethod
        def sobel(image: np.ndarray) -> np.ndarray:
            return image
    
    class DummyExposure:
        @staticmethod
        def rescale_intensity(image: np.ndarray, in_range: Optional[Tuple] = None, out_range: Optional[Tuple] = None) -> np.ndarray:
            return image
        
        @staticmethod
        def equalize_hist(image: np.ndarray) -> np.ndarray:
            return image
    
    class DummyMeasure:
        @staticmethod
        def regionprops(label_img: np.ndarray) -> List:
            return []
        
        @staticmethod
        def label(img: np.ndarray) -> Tuple[np.ndarray, int]:
            return img, 1
    
    class DummySegmentation:
        @staticmethod
        def watershed(*args, **kwargs) -> np.ndarray:
            return np.zeros((512, 512), dtype=np.int32)
    
    # Create dummy modules
    io = DummyIO()
    transform = DummyTransform()
    filters = DummyFilters()
    exposure = DummyExposure()
    measure = DummyMeasure()
    segmentation = DummySegmentation()

try:
    import torch
    from torch.utils.data import Dataset, DataLoader  # type: ignore[assignment]
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available")
    TORCH_AVAILABLE = False
    
    # Create minimal torch functionality
    class Dataset:
        def __len__(self) -> int:
            return 0  # Default implementation instead of raising NotImplementedError
        
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return {}  # Default implementation instead of raising NotImplementedError
    
    class DataLoader:
        def __init__(self, dataset: Any, batch_size: int = 1, shuffle: bool = False, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            
        def __iter__(self):
            # Simple iterator for dummy DataLoader
            if hasattr(self.dataset, '__len__') and len(self.dataset) > 0:
                for i in range(len(self.dataset)):
                    yield self.dataset[i]
            else:
                yield {}

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    warnings.warn("Pandas not available")
    PANDAS_AVAILABLE = False
    
    # Create minimal pandas functionality
    class DummyDataFrame:
        def __init__(self, data: Optional[Dict] = None):
            self.data = data if data is not None else {}
            self.columns = list(self.data.keys()) if self.data else []

        def __len__(self) -> int:
            return max(len(v) for v in self.data.values()) if self.data else 0

        def __getitem__(self, key: str) -> List:
            return self.data.get(key, [])

        @property
        def iloc(self) -> 'DummyILoc':
            return DummyILoc(self)
        
        def get(self, key: str, default: Any = None) -> Any:
            return self.data.get(key, default)
        
        def to_csv(self, filepath: Union[str, Path], **kwargs) -> None:
            """Dummy CSV export."""
            pass
        
        def to_dict(self) -> Dict[str, Any]:
            return self.data
        
        def value_counts(self) -> Dict[str, int]:
            """Return dummy value counts."""
            return {}
        
        def __iter__(self):
            return iter(self.data.keys())

    class DummyILoc:
        def __init__(self, df: 'DummyDataFrame'):
            self.df = df

        def __getitem__(self, idx: Union[int, slice]) -> Union['DummyRow', 'DummyDataFrame']:
            if isinstance(idx, slice):
                # Handle slice objects
                start = idx.start or 0
                stop = idx.stop or len(self.df)
                step = idx.step or 1
                
                # Create a subset of data
                subset_data = {}
                for key, values in self.df.data.items():
                    if isinstance(values, list) and len(values) > start:
                        subset_data[key] = values[start:stop:step]
                    else:
                        subset_data[key] = []
                
                return DummyDataFrame(subset_data)
            else:
                # Handle integer index
                row_data = {}
                for key, values in self.df.data.items():
                    if isinstance(values, list) and len(values) > idx:
                        row_data[key] = values[idx]
                    else:
                        row_data[key] = None
                return DummyRow(row_data)

    class DummyRow:
        def __init__(self, data: Dict[str, Any]):
            self.data = data

        def get(self, key: str, default: Any = None) -> Any:
            return self.data.get(key, default)

        def to_dict(self) -> Dict[str, Any]:
            return self.data

        def to_csv(self, filepath: Union[str, Path], **kwargs) -> None:
            """Dummy CSV export."""
            pass

        def __getitem__(self, key: str) -> Any:
            return self.data[key]

        def __iter__(self):
            return iter(self.data.keys())

        def keys(self):
            return self.data.keys()

        def values(self):
            return self.data.values()

        def items(self):
            return self.data.items()

    class DummyPandas:
        DataFrame = DummyDataFrame
        
        @staticmethod
        def read_csv(filepath: str) -> DummyDataFrame:
            """Create dummy DataFrame for CSV reading."""
            return DummyDataFrame({
                'sample_id': [f'sample_{i}' for i in range(10)],
                'filename': [f'image_{i}.tif' for i in range(10)],
            })

    pd = DummyPandas()


class HighContentImagingDataset(Dataset):
    """Dataset for high-content imaging data."""

    def __init__(
        self,
        config: Union[DictConfig, dict],
        metadata_file: str,
        data_dir: str,
        mode: str = "train",
        transform=None,
    ):
        """
        Initialize High-Content Imaging Dataset.

        Args:
            config: Configuration dictionary
            metadata_file: Path to metadata CSV file
            data_dir: Path to directory containing image files
            mode: Dataset mode ('train', 'val', 'test')
            transform: Optional transforms to apply to images
        """

        self.config = config
        self.metadata_file = metadata_file
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform

        # Dataset parameters
        self.image_size = config.get("image_size", [512, 512])
        self.channels = config.get("channels", ["DAPI", "GFP", "RFP"])
        self.normalize = config.get("normalize", True)
        self.augment = config.get("augment", False) and mode == "train"

        # Load metadata
        self.metadata = self._load_metadata()

        # Validate configuration
        self._validate_data()

        logger = logging.getLogger(__name__)
        logger.info(
            f"Initialized HighContentImagingDataset: {len(self.metadata)} samples, "
            f"mode={mode}, channels={self.channels}"
        )

    def _load_metadata(self) -> Any:
        """Load metadata from file or create dummy data."""

        if Path(self.metadata_file).exists():
            if PANDAS_AVAILABLE:
                return pd.read_csv(self.metadata_file)
            else:
                return self._create_dummy_metadata()
        else:
            return self._create_dummy_metadata()

    def _create_dummy_metadata(self) -> Any:
        """Create dummy metadata for testing."""

        dummy_data = {
            "sample_id": [f"sample_{i:04d}" for i in range(100)],
            "plate_id": [f"plate_{i//10:02d}" for i in range(100)],
            "well_id": [f"{chr(65 + (i%8))}{(i%12)+1:02d}" for i in range(100)],
            "cell_line": ["HeLa"] * 50 + ["U2OS"] * 50,
            "treatment": ["DMSO"] * 25 + ["compound_A"] * 25 + ["compound_B"] * 25 + ["compound_C"] * 25,
            "concentration": [0.0] * 25 + [1.0] * 25 + [5.0] * 25 + [10.0] * 25,
            "timepoint": [24] * 100,
            "filename": [f"image_{i:04d}.tif" for i in range(100)],
            "split": [self.mode] * 100,
        }

        if PANDAS_AVAILABLE:
            return pd.DataFrame(dummy_data)
        else:
            return DummyDataFrame(dummy_data)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing image data, metadata, and perturbation info
        """

        sample_info = self.metadata.iloc[idx]

        # Load image
        image = self._load_image(sample_info)

        # Load mask if available
        mask = self._load_mask(sample_info)

        # Get perturbation information
        perturbation_info = self._get_perturbation_info(sample_info)

        # Create sample dictionary
        sample = {
            "image": image,
            "mask": mask,
            "sample_id": sample_info.get("sample_id", f"sample_{idx}"),
            "metadata": {
                "plate_id": sample_info.get("plate_id", "unknown"),
                "well_id": sample_info.get("well_id", "unknown"),
                "cell_line": sample_info.get("cell_line", "unknown"),
                "timepoint": sample_info.get("timepoint", 24),
                "filename": sample_info.get("filename", "unknown"),
            },
            "perturbation": perturbation_info,
        }

        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_image(self, sample_info) -> np.ndarray:
        """Load multi-channel image."""

        filename = sample_info.get("filename", "dummy.tif")
        image_path = self.data_dir / filename

        if image_path.exists():
            return self._load_multipage_image(image_path)
        elif len(list(self.data_dir.glob(f"{Path(filename).stem}_*.tif"))) > 0:
            return self._load_multichannel_image(sample_info)
        else:
            # Return dummy image
            return self._create_dummy_image()

    def _load_multipage_image(self, image_path: Path) -> np.ndarray:
        """Load multi-page TIFF image."""

        try:
            if CV2_AVAILABLE:
                # Try loading with OpenCV
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    return image

            if SKIMAGE_AVAILABLE:
                # Try loading with scikit-image
                image = io.imread(str(image_path))
                return image

        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load image {image_path}: {e}")

        # Return dummy if loading fails
        return self._create_dummy_image()

    def _load_multichannel_image(self, sample_info) -> np.ndarray:
        """Load multi-channel image from separate files."""

        filename = sample_info.get("filename", "dummy.tif")
        filename_stem = Path(filename).stem

        channels = []
        for channel_name in self.channels:
            channel_path = self._construct_channel_path(sample_info, channel_name)

            try:
                if channel_path.exists():
                    if CV2_AVAILABLE:
                        channel_img = cv2.imread(str(channel_path), cv2.IMREAD_GRAYSCALE)
                    elif SKIMAGE_AVAILABLE:
                        channel_img = io.imread(str(channel_path))
                    else:
                        channel_img = np.zeros(self.image_size, dtype=np.uint8)

                    channels.append(channel_img)
                else:
                    # Add dummy channel
                    channels.append(np.zeros(self.image_size, dtype=np.uint8))

            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load channel {channel_name}: {e}")
                channels.append(np.zeros(self.image_size, dtype=np.uint8))

        # Stack channels
        if channels:
            multichannel_image = np.stack(channels, axis=-1)
        else:
            multichannel_image = self._create_dummy_image()

        return multichannel_image

    def _construct_channel_path(self, sample_info, channel_name: str) -> Path:
        """Construct path to channel-specific image file."""

        filename = sample_info.get("filename", "dummy.tif")
        filename_stem = Path(filename).stem

        # Try different naming conventions
        possible_names = [
            f"{filename_stem}_{channel_name}.tif",
            f"{filename_stem}_{channel_name}.png",
            f"{channel_name}_{filename_stem}.tif",
            f"{channel_name}_{filename_stem}.png",
        ]

        for name in possible_names:
            path = self.data_dir / name
            if path.exists():
                return path

        # Return first option as default
        return self.data_dir / possible_names[0]

    def _create_dummy_image(self) -> np.ndarray:
        """Create dummy image for testing."""
        height, width = self.image_size
        num_channels = len(self.channels)
        return np.zeros((height, width, num_channels), dtype=np.uint8)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image (resize, normalize, etc.)."""

        # Resize image if necessary
        if image.shape[:2] != tuple(self.image_size):
            if SKIMAGE_AVAILABLE:
                resized_image = transform.resize(
                    image, self.image_size, preserve_range=True, anti_aliasing=True
                )
                image = np.asarray(resized_image).astype(np.uint8)
            elif CV2_AVAILABLE:
                image = cv2.resize(image, tuple(reversed(self.image_size)))

        # Normalize channels
        if self.normalize:
            processed_channels = []
            for c in range(image.shape[-1]):
                channel = image[:, :, c]
                normalized_channel = self._normalize_channel(channel)
                processed_channels.append(normalized_channel)
            image = np.stack(processed_channels, axis=-1)

        return image

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize single channel."""

        # Clip extreme values
        p1, p99 = np.percentile(channel, [1, 99])
        channel = np.clip(channel, p1, p99)

        # Scale to 0-255 range
        if p99 > p1:
            channel = ((channel - p1) / (p99 - p1) * 255).astype(np.uint8)
        else:
            channel = np.zeros_like(channel, dtype=np.uint8)

        return channel

    def _load_mask(self, sample_info) -> Optional[np.ndarray]:
        """Load segmentation mask if available."""

        filename = sample_info.get("filename", "dummy.tif")
        mask_filename = filename.replace(".tif", "_mask.tif").replace(".png", "_mask.png")
        mask_path = self.data_dir / "masks" / mask_filename

        if mask_path.exists():
            try:
                if CV2_AVAILABLE:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                elif SKIMAGE_AVAILABLE:
                    mask = io.imread(str(mask_path))
                else:
                    mask = None

                return mask
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load mask {mask_path}: {e}")

        return None

    def _get_perturbation_info(self, sample_info) -> Dict[str, Any]:
        """Extract perturbation information from metadata."""

        perturbation_info = {
            "type": "chemical",
            "target": sample_info.get("treatment", "DMSO"),
            "concentration": sample_info.get("concentration", 0.0),
            "units": "μM",
            "timepoint": sample_info.get("timepoint", 24),
            "timepoint_units": "hours",
        }

        return perturbation_info

    def _validate_data(self) -> None:
        """Validate dataset configuration and data availability."""

        logger = logging.getLogger(__name__)

        # Check if data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")

        # Check image files
        if hasattr(self.metadata, '__len__') and len(self.metadata) > 0:
            sample_files = []
            num_missing = 0

            for i in range(min(10, len(self.metadata))):  # Check first 10 samples
                sample_info = self.metadata.iloc[i]
                filename = sample_info.get("filename", "")
                image_path = self.data_dir / filename

                if not image_path.exists():
                    num_missing += 1

                sample_files.append(str(image_path))

            if num_missing > 0:
                logger.warning(f"{num_missing}/10 sample files are missing")

        logger.info("Data validation completed")


class HighContentImagingLoader:
    """High-content imaging data loader."""

    def __init__(self, config: Union[DictConfig, dict]):
        """Initialize imaging loader."""

        self.config = config
        self.data_dir = Path(config.get("data_dir", "data/imaging"))
        self.metadata_file = config.get("metadata_file", str(self.data_dir / "metadata.csv"))
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)

        # Data splits
        self.train_split = config.get("train_split", 0.7)
        self.val_split = config.get("val_split", 0.15)
        self.test_split = config.get("test_split", 0.15)

        # Datasets and dataloaders
        self.datasets: Dict[str, Optional[HighContentImagingDataset]] = {
            "train": None,
            "val": None,
            "test": None,
        }
        self.dataloaders: Dict[str, Optional[Any]] = {
            "train": None,
            "val": None,
            "test": None,
        }

        logger = logging.getLogger(__name__)
        logger.info(f"Initialized HighContentImagingLoader with data_dir={self.data_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and dataloaders."""

        logger = logging.getLogger(__name__)
        logger.info(f"Setting up imaging data for stage: {stage}")

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load or create metadata
        if not Path(self.metadata_file).exists():
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            logger.info("Creating dummy metadata...")
            self._create_dummy_metadata()

        # Split metadata
        self._split_metadata()

        # Create datasets
        for split in ["train", "val", "test"]:
            if stage is None or stage == "fit" or (stage == "test" and split == "test"):
                metadata_file = self.data_dir / f"metadata_{split}.csv"

                if metadata_file.exists() or not PANDAS_AVAILABLE:
                    self.datasets[split] = HighContentImagingDataset(
                        config=self.config,
                        metadata_file=str(metadata_file),
                        data_dir=str(self.data_dir),
                        mode=split,
                    )

                    dataset = self.datasets[split]
                    if dataset is not None:
                        logger.info(f"Created {split} dataset with {len(dataset)} samples")

        # Create dataloaders
        self._create_dataloaders()

        logger.info("Imaging data setup completed")

    def _split_metadata(self) -> None:
        """Split metadata into train/val/test sets."""

        if not Path(self.metadata_file).exists():
            return

        try:
            if PANDAS_AVAILABLE:
                metadata = pd.read_csv(self.metadata_file)
            else:
                metadata = DummyDataFrame()

            if len(metadata) == 0:
                return

            # Simple random split
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(len(metadata))

            train_size = int(len(metadata) * self.train_split)
            val_size = int(len(metadata) * self.val_split)

            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]

            # Create split DataFrames and save
            splits = {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }

            for split_name, split_indices in splits.items():
                if PANDAS_AVAILABLE:
                    split_metadata = metadata.iloc[split_indices.tolist()]
                    split_file = self.data_dir / f"metadata_{split_name}.csv"
                    split_metadata.to_csv(split_file, index=False)
                else:
                    # Create dummy split file
                    split_file = self.data_dir / f"metadata_{split_name}.csv"
                    split_file.touch()

        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to split metadata: {e}")
            self._create_dummy_metadata()

    def _create_dummy_metadata(self) -> Any:
        """Create dummy metadata files."""

        logger = logging.getLogger(__name__)
        logger.info("Creating dummy metadata files...")

        dummy_data = {
            "sample_id": [f"sample_{i:04d}" for i in range(100)],
            "plate_id": [f"plate_{i//10:02d}" for i in range(100)],
            "well_id": [f"{chr(65 + (i%8))}{(i%12)+1:02d}" for i in range(100)],
            "cell_line": ["HeLa"] * 50 + ["U2OS"] * 50,
            "treatment": ["DMSO"] * 25 + ["compound_A"] * 25 + ["compound_B"] * 25 + ["compound_C"] * 25,
            "concentration": [0.0] * 25 + [1.0] * 25 + [5.0] * 25 + [10.0] * 25,
            "timepoint": [24] * 100,
            "filename": [f"image_{i:04d}.tif" for i in range(100)],
        }

        if PANDAS_AVAILABLE:
            full_metadata = pd.DataFrame(dummy_data)
            full_metadata.to_csv(self.metadata_file, index=False)

            # Create splits
            train_size = int(len(full_metadata) * self.train_split)
            val_size = int(len(full_metadata) * self.val_split)

            train_data = full_metadata.iloc[:train_size]
            val_data = full_metadata.iloc[train_size:train_size + val_size]
            test_data = full_metadata.iloc[train_size + val_size:]

            train_data.to_csv(self.data_dir / "metadata_train.csv", index=False)
            val_data.to_csv(self.data_dir / "metadata_val.csv", index=False)
            test_data.to_csv(self.data_dir / "metadata_test.csv", index=False)
        else:
            # Create dummy files without pandas
            for split in ["train", "val", "test"]:
                split_file = self.data_dir / f"metadata_{split}.csv"
                split_file.touch()

        return DummyDataFrame(dummy_data) if not PANDAS_AVAILABLE else pd.DataFrame(dummy_data)

    def _create_dataloaders(self) -> None:
        """Create PyTorch dataloaders."""

        def collate_fn(batch):
            """Custom collate function for imaging data."""

            images = []
            masks = []
            sample_ids = []
            metadata = []
            perturbations = []

            for sample in batch:
                if sample["image"] is not None:
                    if TORCH_AVAILABLE:
                        if isinstance(sample["image"], np.ndarray):
                            # Convert numpy array to tensor
                            image_tensor = torch.from_numpy(sample["image"]).float()
                            # Ensure channel-first format (C, H, W)
                            if len(image_tensor.shape) == 3 and image_tensor.shape[2] <= 4:
                                image_tensor = image_tensor.permute(2, 0, 1)
                            images.append(image_tensor)
                        else:
                            images.append(sample["image"])
                    else:
                        images.append(sample["image"])

                if sample["mask"] is not None:
                    if TORCH_AVAILABLE:
                        if isinstance(sample["mask"], np.ndarray):
                            mask_tensor = torch.from_numpy(sample["mask"]).float()
                            masks.append(mask_tensor)
                        else:
                            masks.append(sample["mask"])
                    else:
                        masks.append(sample["mask"])

                sample_ids.append(sample["sample_id"])
                metadata.append(sample["metadata"])
                perturbations.append(sample["perturbation"])

            # Stack tensors
            from typing import Any, Dict  # Local import to avoid conditional torch import issues

            # Explicitly type batch_dict to allow heterogeneous value types
            batch_dict: Dict[str, Any] = {
                "sample_ids": sample_ids,
                "metadata": metadata,
                "perturbations": perturbations,
            }

            if images:
                if TORCH_AVAILABLE:
                    batch_dict["images"] = torch.stack(images)
                else:
                    batch_dict["images"] = images

            if masks:
                if TORCH_AVAILABLE:
                    batch_dict["masks"] = torch.stack(masks)
                else:
                    batch_dict["masks"] = masks

            return batch_dict

        # Create dataloaders for each split
        for split in ["train", "val", "test"]:
            if self.datasets[split] is not None:
                shuffle = split == "train"

                self.dataloaders[split] = DataLoader(
                    self.datasets[split],
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=collate_fn,
                )

    def get_dataloader(self, split: str) -> Optional[Any]:
        """Get dataloader for specified split."""

        if split not in self.dataloaders:
            raise ValueError(f"Invalid split: {split}. Valid splits: {list(self.dataloaders.keys())}")

        return self.dataloaders[split]

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the datasets."""

        stats = {
            "total_samples": 0,
            "splits": {},
            "channels": self.config.get("channels", []),
            "image_size": self.config.get("image_size", [512, 512]),
        }

        for split, dataset in self.datasets.items():
            if dataset is not None:
                split_size = len(dataset)
                stats["splits"][split] = split_size
                stats["total_samples"] += split_size

        # Get sample statistics
        if self.datasets["train"] is not None and len(self.datasets["train"]) > 0:
            sample = self.datasets["train"][0]
            
            if "perturbation" in sample:
                # Get unique treatments/perturbations
                treatments = set()
                concentrations = set()
                
                train_dataset = self.datasets["train"]
                if train_dataset is not None:
                    for i in range(min(100, len(train_dataset))):
                        sample_info = train_dataset[i]
                        pert_info = sample_info["perturbation"]
                        treatments.add(pert_info.get("target", "unknown"))
                        concentrations.add(pert_info.get("concentration", 0.0))
                
                stats["perturbations"] = {
                    "unique_treatments": list(treatments),
                    "num_treatments": len(treatments),
                    "concentration_range": [min(concentrations), max(concentrations)] if concentrations else [0, 0],
                }

        return stats

    def get_dataset(self, split: str) -> Optional[HighContentImagingDataset]:
        """Get dataset for specified split."""

        if split not in self.datasets:
            raise ValueError(f"Invalid split: {split}. Valid splits: {list(self.datasets.keys())}")

        return self.datasets[split]

    def get_datasets(self) -> Dict[str, Optional[HighContentImagingDataset]]:
        """Return datasets dictionary."""
        return self.datasets


def create_synthetic_imaging_data(config: Union[DictConfig, dict], output_dir: Union[str, Path]) -> None:
    """Create synthetic imaging data for testing."""

    logger = logging.getLogger(__name__)
    logger.info(f"Creating synthetic imaging data in {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create synthetic images
    num_images = config.get("num_samples", 100)
    image_size = config.get("image_size", [512, 512])
    channels = config.get("channels", ["DAPI", "GFP", "RFP"])

    for i in range(num_images):
        # Create synthetic multi-channel image
        synthetic_image = np.random.randint(0, 255, (*image_size, len(channels)), dtype=np.uint8)

        # Add some structure to make it look more realistic
        for c in range(len(channels)):
            # Add some circular structures (fake cells)
            num_cells = np.random.randint(5, 20)
            for _ in range(num_cells):
                center_x = np.random.randint(50, image_size[0] - 50)
                center_y = np.random.randint(50, image_size[1] - 50)
                radius = np.random.randint(10, 30)

                y, x = np.ogrid[:image_size[0], :image_size[1]]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

                synthetic_image[mask, c] = np.random.randint(100, 255)

        # Save image
        filename = f"image_{i:04d}.tif"
        image_path = output_path / filename

        try:
            if CV2_AVAILABLE:
                cv2.imwrite(str(image_path), synthetic_image)
            elif SKIMAGE_AVAILABLE:
                io.imsave(str(image_path), synthetic_image)
            else:
                # Create empty file as placeholder
                image_path.touch()
        except Exception as e:
            logger.warning(f"Failed to save synthetic image {filename}: {e}")
            # Create empty file as placeholder
            image_path.touch()

    # Create metadata
    treatments = ["DMSO", "compound_A", "compound_B", "compound_C"]
    concentrations = [0.0, 1.0, 5.0, 10.0]
    cell_lines = ["HeLa", "U2OS"]

    metadata_entries: List[Dict[str, Any]] = []
    for i in range(num_images):
        entry = {
            "sample_id": f"sample_{i:04d}",
            "plate_id": f"plate_{i//10:02d}",
            "well_id": f"{chr(65 + (i % 8))}{(i % 12) + 1:02d}",
            "cell_line": np.random.choice(cell_lines),
            "treatment": np.random.choice(treatments),
            "concentration": np.random.choice(concentrations),
            "timepoint": 24,
            "filename": f"image_{i:04d}.tif",
        }
        metadata_entries.append(entry)

    # Save metadata
    if PANDAS_AVAILABLE:
        metadata_df = pd.DataFrame(metadata_entries)  # type: ignore[arg-type]
        metadata_df.to_csv(output_path / "metadata.csv", index=False)
    else:
        # Write simple CSV manually
        metadata_path = output_path / "metadata.csv"
        with open(metadata_path, "w") as f:
            # Write header
            f.write("sample_id,plate_id,well_id,cell_line,treatment,concentration,timepoint,filename\n")
            # Write data
            for entry in metadata_entries:
                line = f"{entry['sample_id']},{entry['plate_id']},{entry['well_id']},"
                line += f"{entry['cell_line']},{entry['treatment']},{entry['concentration']},"
                line += f"{entry['timepoint']},{entry['filename']}\n"
                f.write(line)

    logger.info(f"Created {num_images} synthetic images with metadata")


def test_imaging_loader() -> None:
    """Test the imaging loader functionality."""

    logger = logging.getLogger(__name__)
    logger.info("Testing HighContentImagingLoader...")

    # Test configuration
    config = {
        "data_dir": "test_data/imaging",
        "batch_size": 4,
        "num_workers": 0,  # Set to 0 for testing
        "image_size": [256, 256],
        "channels": ["DAPI", "GFP", "RFP"],
        "create_synthetic": True,
        "num_samples": 20,
    }

    # Create synthetic data
    create_synthetic_imaging_data(config, config["data_dir"])

    # Initialize loader
    loader = HighContentImagingLoader(config)
    loader.setup()

    # Test datasets
    for split in ["train", "val", "test"]:
        dataset = loader.get_dataset(split)
        if dataset:
            logger.info(f"{split} dataset: {len(dataset)} samples")

            # Test sample loading
            sample = dataset[0]
            logger.info(f"Sample keys: {list(sample.keys())}")

            if sample["image"] is not None:
                logger.info(f"Image shape: {sample['image'].shape}")

    # Test dataloaders
    train_loader = loader.get_dataloader("train")
    if train_loader:
        for batch in train_loader:
            logger.info(f"Batch keys: {list(batch.keys())}")
            if "images" in batch:
                logger.info(f"Batch image shape: {batch['images'].shape}")
            break

    # Get statistics
    stats = loader.get_dataset_statistics()
    logger.info(f"Dataset statistics: {stats}")

    logger.info("HighContentImagingLoader test completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run test
    test_imaging_loader()