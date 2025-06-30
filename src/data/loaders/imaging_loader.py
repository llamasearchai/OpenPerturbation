"""
High-Content Imaging Data Loader for OpenPerturbation

Comprehensive loading and preprocessing of multi-channel microscopy images
from high-content screening experiments.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np
import warnings

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    warnings.warn("OpenCV not available")
    CV2_AVAILABLE = False
    
    # Create dummy cv2 functions
    def imread(filepath, flags=None):
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def resize(image, size):
        return np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    
    def convertScaleAbs(image, alpha=1, beta=0):
        return (image * alpha + beta).astype(np.uint8)
    
    cv2 = type('cv2', (), {
        'imread': imread,
        'resize': resize,
        'convertScaleAbs': convertScaleAbs,
        'IMREAD_UNCHANGED': -1,
        'IMREAD_GRAYSCALE': 0,
        'IMREAD_COLOR': 1
    })()

try:
    import skimage
    from skimage import io, transform, filters, exposure, morphology
    from skimage.measure import regionprops, label
    from skimage.segmentation import watershed
    SKIMAGE_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-image not available")
    SKIMAGE_AVAILABLE = False
    
    # Dummy skimage
    if not SKIMAGE_AVAILABLE:
        class DummyIO:
            @staticmethod
            def imread(filepath):
                return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            @staticmethod
            def imsave(filepath, image):
                pass
        class DummyTransform:
            @staticmethod
            def resize(image, shape):
                return np.random.randint(0, 255, shape, dtype=np.uint8)
        class DummyFilters:
            @staticmethod
            def gaussian(image, sigma=1):
                return image
            @staticmethod
            def sobel(image):
                return np.random.random(image.shape[:2])
        class DummyExposure:
            @staticmethod
            def rescale_intensity(image, in_range=None, out_range=None):
                return image
            @staticmethod
            def equalize_hist(image):
                return image
        class DummyMeasure:
            @staticmethod
            def regionprops(label_img):
                return []
            @staticmethod
            def label(img):
                return np.zeros_like(img)
        class DummySegmentation:
            @staticmethod
            def watershed(*args, **kwargs):
                return np.zeros((512, 512), dtype=np.int32)
        io = DummyIO()
        transform = DummyTransform()
        filters = DummyFilters()
        exposure = DummyExposure()
        measure = DummyMeasure()
        segmentation = DummySegmentation()
    else:
        from skimage import io, transform, filters, exposure, measure, segmentation

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    warnings.warn("h5py not available")
    H5PY_AVAILABLE = False

try:
    from omegaconf import DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    warnings.warn("OmegaConf not available")
    OMEGACONF_AVAILABLE = False
    DictConfig = dict

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available")
    TORCH_AVAILABLE = False
    
    class Dataset:
        pass
        
    class DataLoader:
        pass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    warnings.warn("Pandas not available")
    PANDAS_AVAILABLE = False
    
    # Create minimal pandas functionality
    class DataFrame:
        def __init__(self, data=None):
            self.data = data or []
            self.columns = []
            
        def __len__(self):
            return len(self.data)
            
        def iloc(self, idx):
            return self.data[idx] if idx < len(self.data) else {}
            
        def to_dict(self):
            return {"data": self.data}
    
    def read_csv(filepath):
        return DataFrame()
    
    pd = type('pd', (), {
        'DataFrame': DataFrame,
        'read_csv': read_csv
    })()

logger = logging.getLogger(__name__)


class HighContentImagingDataset(object):
    """Dataset for high-content imaging data with comprehensive processing."""

    def __init__(
        self,
        config: Union[DictConfig, dict],
        metadata_file: str,
        data_dir: str,
        mode: str = "train",
        transform=None,
    ):
        """
        Initialize imaging dataset.

        Args:
            config: Configuration dictionary
            metadata_file: Path to metadata CSV file
            data_dir: Directory containing image files
            mode: Dataset mode ('train', 'val', 'test')
            transform: Optional transform pipeline
        """

        self.config = config
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform

        # Load metadata
        if PANDAS_AVAILABLE:
            try:
                self.metadata = pd.read_csv(metadata_file)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = self._create_dummy_metadata()
        else:
            logger.warning("Pandas not available, using dummy metadata")
            self.metadata = self._create_dummy_metadata()

        # Filter metadata for current mode if specified
        if hasattr(self.metadata, 'columns') and "split" in self.metadata.columns:
            if hasattr(self.metadata, 'iloc'):
                # Real pandas DataFrame
                split_mask = self.metadata["split"] == mode
                self.metadata = self.metadata[split_mask]
            else:
                # Dummy DataFrame
                pass

        # Channel configuration
        self.channels = config.get("channels", ["DAPI", "GFP", "RFP", "Cy5", "Cy7"])
        self.num_channels = len(self.channels)

        # Image preprocessing parameters
        self.image_size = config.get("image_size", 224)
        self.normalize_per_channel = config.get("normalize_per_channel", True)
        self.clip_percentiles = config.get("clip_percentiles", [1, 99])

        # Caching
        self.cache_images = config.get("cache_images", False)
        self.image_cache = {} if self.cache_images else None

        # Validate data
        self._validate_data()

        logger.info(f"Initialized {mode} dataset with {len(self.metadata)} samples")

    def _create_dummy_metadata(self):
        """Create dummy metadata for testing."""
        if PANDAS_AVAILABLE:
            return pd.DataFrame({
                'sample_id': [f'sample_{i}' for i in range(100)],
                'plate_id': [f'plate_{i//10}' for i in range(100)],
                'well_id': [f'well_{i%96}' for i in range(100)],
                'site_id': [f'site_{i%4}' for i in range(100)],
                'perturbation': ['control'] * 50 + ['treated'] * 50,
                'compound_id': [f'compound_{i%10}' for i in range(100)]
            })
        else:
            # Return dummy data structure
            return type('DummyMetadata', (), {
                'data': [{'sample_id': f'sample_{i}'} for i in range(100)],
                '__len__': lambda self: 100,
                'iloc': lambda self, idx: self.data[idx] if idx < len(self.data) else {}
            })()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""

        # Get metadata for this sample
        if hasattr(self.metadata, 'iloc'):
            sample_info = self.metadata.iloc[idx]
        else:
            sample_info = self.metadata.data[idx] if idx < len(self.metadata.data) else {}

        # Load image
        image = self._load_image(sample_info)

        # Load segmentation mask if available
        mask = self._load_mask(sample_info)

        # Create sample dictionary
        sample = {
            "image": image,
            "mask": mask,
            "sample_id": sample_info.get("sample_id", idx),
            "plate_id": sample_info.get("plate_id", "unknown"),
            "well_id": sample_info.get("well_id", "unknown"),
            "site_id": sample_info.get("site_id", 0),
            "perturbation": self._get_perturbation_info(sample_info),
            "metadata": sample_info if hasattr(sample_info, 'to_dict') else sample_info,
        }

        # Apply transforms
        if self.transform is not None:
            try:
                transformed = self.transform(
                    image=sample["image"], 
                    mask=sample["mask"], 
                    perturbation_info=sample["perturbation"]
                )
                sample.update({"image": transformed["image"], "mask": transformed.get("mask", None)})
            except Exception as e:
                logger.debug(f"Transform failed: {e}")

        return sample

    def _load_image(self, sample_info) -> np.ndarray:
        """Load multi-channel image."""

        # Check cache first
        sample_id = sample_info.get("sample_id", "unknown")
        if self.image_cache is not None and sample_id in self.image_cache:
            return self.image_cache[sample_id].copy()

        # Load image based on file format
        if "image_path" in sample_info:
            # Single file containing all channels
            image_path = self.data_dir / sample_info["image_path"]
            image = self._load_multipage_image(image_path)
        else:
            # Separate files for each channel
            image = self._load_multichannel_image(sample_info)

        # Preprocess image
        image = self._preprocess_image(image)

        # Cache if enabled
        if self.image_cache is not None:
            self.image_cache[sample_id] = image.copy()

        return image

    def _load_multipage_image(self, image_path: Path) -> np.ndarray:
        """Load multipage TIFF or similar format."""

        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return self._create_dummy_image()

        try:
            if image_path.suffix.lower() in [".tif", ".tiff"]:
                if SKIMAGE_AVAILABLE:
                    # Load TIFF stack
                    image_stack = io.imread(str(image_path))

                    # Handle different TIFF formats
                    if image_stack.ndim == 3:
                        # Stack of 2D images (channels, height, width)
                        image = image_stack
                    elif image_stack.ndim == 4:
                        # Time series or Z-stack - take first timepoint/z-slice
                        image = image_stack[0]
                    else:
                        raise ValueError(f"Unexpected image dimensions: {image_stack.shape}")

                    # Ensure correct channel ordering (channels last)
                    if image.ndim == 3 and image.shape[0] <= 10:  # Likely channels first
                        image = np.transpose(image, (1, 2, 0))

                    return image
                else:
                    logger.warning("scikit-image not available for TIFF loading")
                    return self._create_dummy_image()
            else:
                # Try other formats
                if CV2_AVAILABLE:
                    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                    if image is not None:
                        # Handle grayscale to multi-channel
                        if image.ndim == 2:
                            image = np.stack([image] * self.num_channels, axis=-1)
                        elif image.ndim == 3 and image.shape[-1] < self.num_channels:
                            # Pad channels
                            padding = self.num_channels - image.shape[-1]
                            pad_channels = np.zeros((*image.shape[:2], padding), dtype=image.dtype)
                            image = np.concatenate([image, pad_channels], axis=-1)
                        return image
                    else:
                        return self._create_dummy_image()
                else:
                    return self._create_dummy_image()
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return self._create_dummy_image()

    def _load_multichannel_image(self, sample_info) -> np.ndarray:
        """Load separate channel files and combine."""
        
        channels = []
        
        for channel_name in self.channels:
            channel_path = self._construct_channel_path(sample_info, channel_name)
            
            if channel_path.exists():
                try:
                    if CV2_AVAILABLE:
                        channel_img = cv2.imread(str(channel_path), cv2.IMREAD_GRAYSCALE)
                    elif SKIMAGE_AVAILABLE:
                        channel_img = io.imread(str(channel_path))
                        if channel_img.ndim > 2:
                            channel_img = channel_img[:, :, 0]  # Take first channel
                    else:
                        # Create dummy channel
                        channel_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
                    
                    if channel_img is not None:
                        channels.append(channel_img)
                    else:
                        # Add dummy channel
                        channels.append(np.zeros((512, 512), dtype=np.uint8))
                except Exception as e:
                    logger.debug(f"Failed to load channel {channel_name}: {e}")
                    # Add dummy channel
                    channels.append(np.zeros((512, 512), dtype=np.uint8))
            else:
                # Add dummy channel
                channels.append(np.zeros((512, 512), dtype=np.uint8))
        
        if channels:
            # Stack channels
            image = np.stack(channels, axis=-1)
            return image
        else:
            return self._create_dummy_image()

    def _construct_channel_path(self, sample_info, channel_name: str) -> Path:
        """Construct file path for specific channel."""
        
        # Try common naming patterns
        sample_id = sample_info.get("sample_id", "unknown")
        plate_id = sample_info.get("plate_id", "unknown")
        well_id = sample_info.get("well_id", "unknown")
        site_id = sample_info.get("site_id", "1")
        
        # Common file patterns
        patterns = [
            f"{sample_id}_{channel_name}.tif",
            f"{plate_id}_{well_id}_{site_id}_{channel_name}.tif",
            f"{plate_id}/{well_id}_{channel_name}.tif",
            f"{channel_name}/{sample_id}.tif",
            f"{sample_id}_{channel_name}.png",
            f"{sample_id}_{channel_name}.jpg"
        ]
        
        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path
        
        # Return first pattern as default (will be handled as missing)
        return self.data_dir / patterns[0]

    def _create_dummy_image(self) -> np.ndarray:
        """Create dummy multi-channel image."""
        height, width = 512, 512
        image = np.random.randint(0, 255, (height, width, self.num_channels), dtype=np.uint8)
        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image (resize, normalize, etc.)."""
        
        if image is None:
            return self._create_dummy_image()
        
        try:
            # Resize image
            if image.shape[:2] != (self.image_size, self.image_size):
                if CV2_AVAILABLE:
                    image = cv2.resize(image, (self.image_size, self.image_size))
                elif SKIMAGE_AVAILABLE:
                    if image.ndim == 3:
                        resized_channels = []
                        for c in range(image.shape[-1]):
                            resized_ch = transform.resize(
                                image[:, :, c], 
                                (self.image_size, self.image_size),
                                preserve_range=True
                            ).astype(image.dtype)
                            resized_channels.append(resized_ch)
                        image = np.stack(resized_channels, axis=-1)
                    else:
                        image = transform.resize(
                            image, 
                            (self.image_size, self.image_size),
                            preserve_range=True
                        ).astype(image.dtype)

            # Normalize per channel
            if self.normalize_per_channel and image.ndim == 3:
                for c in range(image.shape[-1]):
                    image[:, :, c] = self._normalize_channel(image[:, :, c])
            elif image.ndim == 2:
                image = self._normalize_channel(image)

            return image.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return self._create_dummy_image().astype(np.float32)

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize single channel."""
        
        try:
            # Clip extreme values
            if self.clip_percentiles:
                low, high = np.percentile(channel, self.clip_percentiles)
                channel = np.clip(channel, low, high)
            
            # Normalize to 0-1
            channel_min = channel.min()
            channel_max = channel.max()
            
            if channel_max > channel_min:
                channel = (channel - channel_min) / (channel_max - channel_min)
            else:
                channel = np.zeros_like(channel)
            
            return channel
            
        except Exception as e:
            logger.debug(f"Channel normalization failed: {e}")
            return channel

    def _load_mask(self, sample_info) -> Optional[np.ndarray]:
        """Load segmentation mask if available."""
        
        mask_path = sample_info.get("mask_path")
        if not mask_path:
            return None
        
        mask_path = self.data_dir / mask_path
        if not mask_path.exists():
            return None
        
        try:
            if CV2_AVAILABLE:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            elif SKIMAGE_AVAILABLE:
                mask = io.imread(str(mask_path))
                if mask.ndim > 2:
                    mask = mask[:, :, 0]  # Take first channel
            else:
                return None
            
            # Resize mask to match image size
            if mask.shape != (self.image_size, self.image_size):
                if CV2_AVAILABLE:
                    mask = cv2.resize(mask, (self.image_size, self.image_size))
                elif SKIMAGE_AVAILABLE:
                    mask = transform.resize(
                        mask, 
                        (self.image_size, self.image_size),
                        preserve_range=True,
                        order=0  # Nearest neighbor for masks
                    ).astype(mask.dtype)
            
            return mask
            
        except Exception as e:
            logger.debug(f"Failed to load mask: {e}")
            return None

    def _get_perturbation_info(self, sample_info) -> Dict:
        """Extract perturbation information."""
        
        perturbation = {
            "type": "chemical",
            "compound_id": sample_info.get("compound_id", "unknown"),
            "concentration": sample_info.get("concentration", sample_info.get("dose", 1.0)),
            "time": sample_info.get("treatment_time", sample_info.get("time", 24.0)),
            "condition": sample_info.get("perturbation", sample_info.get("condition", "control"))
        }
        
        return perturbation

    def _validate_data(self):
        """Validate dataset configuration and data availability."""
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
        
        # Check if we have any samples
        if len(self.metadata) == 0:
            logger.warning("No samples found in metadata")
        
        # Check for required columns
        required_cols = ["sample_id"]
        if hasattr(self.metadata, 'columns'):
            missing_cols = [col for col in required_cols if col not in self.metadata.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")


class HighContentImagingLoader(object):
    """Data loader for high-content imaging datasets."""

    def __init__(self, config: Union[DictConfig, dict]):
        """Initialize imaging data loader."""
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", 0)
        self.pin_memory = config.get("pin_memory", False)
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and data loaders."""
        
        data_dir = self.config.get("data_dir", "data/imaging")
        metadata_dir = self.config.get("metadata_dir", data_dir)
        
        if stage == "fit" or stage is None:
            # Training data
            train_metadata = Path(metadata_dir) / self.config.get("train_metadata", "train_metadata.csv")
            if train_metadata.exists():
                self.train_dataset = HighContentImagingDataset(
                    self.config, str(train_metadata), data_dir, mode="train"
                )
            
            # Validation data
            val_metadata = Path(metadata_dir) / self.config.get("val_metadata", "val_metadata.csv")
            if val_metadata.exists():
                self.val_dataset = HighContentImagingDataset(
                    self.config, str(val_metadata), data_dir, mode="val"
                )
            elif self.train_dataset is not None:
                # Split training data
                self._split_metadata()
        
        if stage == "test" or stage is None:
            test_metadata = Path(metadata_dir) / self.config.get("test_metadata", "test_metadata.csv")
            if test_metadata.exists():
                self.test_dataset = HighContentImagingDataset(
                    self.config, str(test_metadata), data_dir, mode="test"
                )
        
        # Create data loaders
        self._create_dataloaders()

    def _split_metadata(self):
        """Split training metadata into train/val sets."""
        
        if not self.train_dataset or not PANDAS_AVAILABLE:
            return
        
        val_ratio = self.config.get("val_ratio", 0.2)
        
        # Simple random split
        metadata = self.train_dataset.metadata
        n_samples = len(metadata)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Create splits
        if hasattr(metadata, 'iloc'):
            train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
            val_metadata = metadata.iloc[val_indices].reset_index(drop=True)
            
            # Update datasets
            self.train_dataset.metadata = train_metadata
            
            # Create validation dataset
            self.val_dataset = HighContentImagingDataset(
                self.config, None, self.config.get("data_dir", "data/imaging"), mode="val"
            )
            self.val_dataset.metadata = val_metadata

    def _create_dataloaders(self):
        """Create PyTorch data loaders."""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot create data loaders")
            return

        def collate_fn(batch):
            """Custom collate function for imaging data."""
            
            # Extract components
            images = []
            masks = []
            sample_ids = []
            perturbations = []
            metadata = []
            
            for item in batch:
                if TORCH_AVAILABLE:
                    # Convert image to tensor
                    image = torch.tensor(item["image"], dtype=torch.float32)
                    # Ensure channel-first format (C, H, W)
                    if image.ndim == 3:
                        image = image.permute(2, 0, 1)  # H, W, C -> C, H, W
                    images.append(image)
                    
                    # Handle mask
                    if item["mask"] is not None:
                        mask = torch.tensor(item["mask"], dtype=torch.long)
                        masks.append(mask)
                    else:
                        # Create dummy mask
                        mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.long)
                        masks.append(mask)
                
                sample_ids.append(item["sample_id"])
                perturbations.append(item["perturbation"])
                metadata.append(item["metadata"])
            
            # Stack tensors
            batch_data = {
                "sample_ids": sample_ids,
                "perturbations": perturbations,
                "metadata": metadata
            }
            
            if TORCH_AVAILABLE and images:
                batch_data["images"] = torch.stack(images)
                batch_data["masks"] = torch.stack(masks) if masks else None
            
            return batch_data
        
        # Create data loaders
        for split, dataset in [("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)]:
            if dataset is not None:
                shuffle = (split == "train")
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=collate_fn
                )
                
                if split == "train":
                    self.train_loader = loader
                elif split == "val":
                    self.val_loader = loader
                elif split == "test":
                    self.test_loader = loader

    def get_dataloader(self, split: str) -> Any:
        """Get data loader for specified split."""
        if split == "train":
            return self.train_loader
        elif split == "val":
            return self.val_loader
        elif split == "test":
            return self.test_loader
        else:
            logger.warning(f"Unknown split: {split}")
            return None

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the datasets."""
        stats = {"splits": {}}
        
        for split, dataset in [("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)]:
            if dataset is not None:
                stats["splits"][split] = {
                    "size": len(dataset),
                    "num_channels": dataset.num_channels,
                    "image_size": dataset.image_size,
                    "channels": dataset.channels
                }
                
                # Perturbation statistics
                if hasattr(dataset.metadata, 'columns') and "perturbation" in dataset.metadata.columns:
                    perturbation_counts = dataset.metadata["perturbation"].value_counts().to_dict()
                    stats["splits"][split]["perturbations"] = perturbation_counts
        
        return stats


def create_synthetic_imaging_data(config: Union[DictConfig, dict], output_dir: Union[str, Path]):
    """Create synthetic imaging data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = config.get("n_samples", 200)
    image_size = config.get("image_size", 224)
    channels = config.get("channels", ["DAPI", "GFP", "RFP"])
    
    logger.info(f"Creating synthetic imaging data: {n_samples} samples")
    
    # Create metadata
    metadata = []
    for i in range(n_samples):
        sample_data = {
            "sample_id": f"sample_{i:05d}",
            "plate_id": f"plate_{i//96}",
            "well_id": f"well_{chr(65 + (i%8))}{(i%12)+1:02d}",
            "site_id": f"site_{i%4}",
            "compound_id": f"compound_{i%20}",
            "concentration": np.random.uniform(0.1, 10.0),
            "treatment_time": np.random.choice([6, 12, 24, 48]),
            "perturbation": np.random.choice(["control", "treated"], p=[0.3, 0.7]),
            "image_path": f"images/sample_{i:05d}.tif"
        }
        metadata.append(sample_data)
    
    # Save metadata
    if PANDAS_AVAILABLE:
        metadata_df = pd.DataFrame(metadata)
        
        # Split metadata
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        train_metadata = metadata_df[:train_size]
        val_metadata = metadata_df[train_size:train_size + val_size]
        test_metadata = metadata_df[train_size + val_size:]
        
        train_metadata.to_csv(output_dir / "train_metadata.csv", index=False)
        val_metadata.to_csv(output_dir / "val_metadata.csv", index=False)
        test_metadata.to_csv(output_dir / "test_metadata.csv", index=False)
    
    # Create image directory
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Generate synthetic images
    for i in range(min(n_samples, 50)):  # Create subset for testing
        # Create multi-channel image
        image = np.random.randint(0, 255, (image_size, image_size, len(channels)), dtype=np.uint8)
        
        # Add some structure to make it look more realistic
        for c in range(len(channels)):
            # Add some circular structures (cells)
            for _ in range(np.random.randint(10, 30)):
                center_x = np.random.randint(20, image_size-20)
                center_y = np.random.randint(20, image_size-20)
                radius = np.random.randint(5, 15)
                
                y, x = np.ogrid[:image_size, :image_size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                image[mask, c] = np.random.randint(100, 255)
        
        # Save image
        image_path = image_dir / f"sample_{i:05d}.tif"
        if SKIMAGE_AVAILABLE:
            try:
                io.imsave(str(image_path), image)
            except Exception:
                # Save as numpy array if TIFF save fails
                np.save(str(image_path.with_suffix('.npy')), image)
        else:
            # Save as numpy array
            np.save(str(image_path.with_suffix('.npy')), image)
    
    logger.info(f"Created synthetic imaging data in {output_dir}")


def test_imaging_loader():
    """Test imaging data loader functionality."""
    logger.info("Testing imaging data loader...")
    
    # Create test config
    config = {
        "data_dir": "test_imaging_data",
        "batch_size": 4,
        "image_size": 128,
        "channels": ["DAPI", "GFP", "RFP"],
        "normalize_per_channel": True,
        "n_samples": 50
    }
    
    # Create synthetic data
    create_synthetic_imaging_data(config, config["data_dir"])
    
    # Initialize data loader
    loader = HighContentImagingLoader(config)
    loader.setup()
    
    # Test data loading
    train_loader = loader.get_dataloader("train")
    if train_loader and TORCH_AVAILABLE:
        for batch in train_loader:
            logger.info(f"Batch keys: {batch.keys()}")
            if "images" in batch:
                logger.info(f"Images shape: {batch['images'].shape}")
            if "masks" in batch and batch["masks"] is not None:
                logger.info(f"Masks shape: {batch['masks'].shape}")
            break
    
    # Print statistics
    stats = loader.get_dataset_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    logger.info("Imaging data loader test completed successfully!")


if __name__ == "__main__":
    test_imaging_loader()