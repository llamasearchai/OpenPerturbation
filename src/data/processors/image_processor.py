"""
Image processing utilities for OpenPerturbations.

This module provides image preprocessing, augmentation, and quality control
for high-content screening data.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import warnings

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False
    warnings.warn("Albumentations not available. Using basic transforms.")

try:
    from skimage import filters, morphology, measure, segmentation
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available. Image processing features may be limited.")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Some image processing features may be limited.")

class CellularImageProcessor:
    """
    Processor for cellular imaging data with biology-aware transformations.
    
    Features:
    - Multi-channel fluorescence normalization
    - Cell segmentation and quality control
    - Biology-aware augmentations
    - Batch processing optimization
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Image parameters
        self.image_size = config.get('image_size', 224)
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5', 'Brightfield'])
        self.target_dtype = config.get('target_dtype', torch.float32)
        
        # Normalization parameters
        self.normalization_method = config.get('normalization_method', 'percentile')
        self.percentile_range = config.get('percentile_range', (1, 99))
        self.global_normalization = config.get('global_normalization', True)
        
        # Segmentation parameters
        self.enable_segmentation = config.get('enable_segmentation', True)
        self.segmentation_channel = config.get('segmentation_channel', 'DAPI')
        self.min_cell_area = config.get('min_cell_area', 100)
        self.max_cell_area = config.get('max_cell_area', 5000)
        
        # Quality control
        self.quality_control = config.get('quality_control', True)
        self.blur_threshold = config.get('blur_threshold', 100)
        self.intensity_threshold = config.get('intensity_threshold', (0.01, 0.99))
        
        # Cache for normalization statistics
        self.normalization_stats = {}
        
        # Initialize augmentation pipeline
        self._setup_augmentations()
    
    def _setup_augmentations(self):
        """Setup augmentation pipelines for training and validation."""
        
        # Training augmentations (biology-aware)
        self.train_transforms = A.Compose([
            # Geometric transformations (preserving cellular structure)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.7
            ),
            
            # Elastic deformation (mild, to simulate natural cell variation)
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=10,
                border_mode=cv2.BORDER_REFLECT,
                p=0.3
            ),
            
            # Optical variations (simulating microscopy artifacts)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Gaussian noise (simulating camera noise)
            A.GaussNoise(
                var_limit=(10, 50),
                p=0.3
            ),
            
            # Resize to target size
            A.Resize(self.image_size, self.image_size),
            
            # Convert to tensor
            ToTensorV2()
        ])
        
        # Validation transforms (minimal)
        self.val_transforms = A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ])
        
        # Test transforms (same as validation)
        self.test_transforms = self.val_transforms
    
    def process_image(self, 
                     image: np.ndarray,
                     metadata: Dict,
                     split: str = 'train',
                     apply_segmentation: bool = None) -> Dict:
        """
        Process a single multi-channel image.
        
        Args:
            image: Multi-channel image array [C, H, W]
            metadata: Image metadata
            split: Dataset split ('train', 'val', 'test')
            apply_segmentation: Whether to apply cell segmentation
            
        Returns:
            Dictionary containing processed image and metadata
        """
        
        # Quality control
        if self.quality_control and not self._passes_quality_control(image):
            return None
        
        # Normalize channels
        normalized_image = self._normalize_image(image, metadata)
        
        # Cell segmentation (if enabled)
        segmentation_mask = None
        cell_features = None
        
        if (apply_segmentation is True or 
            (apply_segmentation is None and self.enable_segmentation)):
            
            segmentation_result = self._segment_cells(normalized_image)
            segmentation_mask = segmentation_result['mask']
            cell_features = segmentation_result['features']
        
        # Apply augmentations
        if split == 'train':
            transform = self.train_transforms
        elif split == 'val':
            transform = self.val_transforms
        else:
            transform = self.test_transforms
        
        # Prepare image for augmentation (convert to HWC format)
        if normalized_image.ndim == 3:
            # Convert from CHW to HWC
            aug_image = np.transpose(normalized_image, (1, 2, 0))
        else:
            aug_image = normalized_image
        
        # Apply transforms
        transformed = transform(image=aug_image.astype(np.float32))
        processed_image = transformed['image']
        
        # Ensure correct tensor format
        if isinstance(processed_image, np.ndarray):
            processed_image = torch.tensor(processed_image, dtype=self.target_dtype)
        
        # Create result dictionary
        result = {
            'image': processed_image,
            'original_shape': image.shape,
            'processed_shape': processed_image.shape,
            'metadata': metadata.copy()
        }
        
        # Add segmentation results if available
        if segmentation_mask is not None:
            result['segmentation_mask'] = torch.tensor(segmentation_mask, dtype=torch.float32)
            result['cell_features'] = cell_features
        
        return result
    
    def _passes_quality_control(self, image: np.ndarray) -> bool:
        """Check if image passes quality control criteria."""
        
        try:
            # Check image dimensions
            if image.ndim != 3 or image.shape[0] != len(self.channels):
                return False
            
            # Check for empty images
            if np.all(image == 0):
                return False
            
            # Check intensity distribution
            for c in range(image.shape[0]):
                channel_img = image[c]
                
                # Check if channel has reasonable intensity range
                channel_min, channel_max = np.percentile(channel_img, [5, 95])
                if channel_max - channel_min < 10:  # Too low dynamic range
                    continue  # Skip this channel but don't fail the image
                
                # Check for saturation
                saturation_fraction = np.mean(channel_img >= np.max(channel_img) * 0.98)
                if saturation_fraction > 0.1:  # More than 10% saturated
                    return False
            
            # Blur detection on nuclear channel (typically DAPI)
            if self.segmentation_channel in self.channels:
                nuclear_idx = self.channels.index(self.segmentation_channel)
                nuclear_img = image[nuclear_idx]
                
                # Laplacian variance (measure of blur)
                laplacian_var = cv2.Laplacian(nuclear_img.astype(np.uint8), cv2.CV_64F).var()
                if laplacian_var < self.blur_threshold:
                    return False
            
            return True
            
        except Exception as e:
            print(f"WARNING: Quality control error: {e}")
            return False
    
    def _normalize_image(self, image: np.ndarray, metadata: Dict) -> np.ndarray:
        """Normalize multi-channel image."""
        
        normalized_image = np.zeros_like(image, dtype=np.float32)
        
        for c, channel_name in enumerate(self.channels):
            channel_img = image[c].astype(np.float32)
            
            # Get normalization statistics
            if self.global_normalization and channel_name in self.normalization_stats:
                # Use global statistics
                stats = self.normalization_stats[channel_name]
                vmin, vmax = stats['percentile_min'], stats['percentile_max']
            else:
                # Compute per-image statistics
                if self.normalization_method == 'percentile':
                    vmin, vmax = np.percentile(channel_img, self.percentile_range)
                elif self.normalization_method == 'minmax':
                    vmin, vmax = np.min(channel_img), np.max(channel_img)
                elif self.normalization_method == 'zscore':
                    # Z-score normalization
                    mean_val = np.mean(channel_img)
                    std_val = np.std(channel_img) + 1e-8
                    normalized_image[c] = (channel_img - mean_val) / std_val
                    continue
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            # Apply normalization
            if vmax > vmin:
                normalized_image[c] = np.clip((channel_img - vmin) / (vmax - vmin), 0, 1)
            else:
                normalized_image[c] = channel_img
        
        return normalized_image
    
    def _segment_cells(self, image: np.ndarray) -> Dict:
        """Segment cells in the image."""
        
        try:
            # Get nuclear channel
            if self.segmentation_channel not in self.channels:
                return {'mask': None, 'features': None}
            
            nuclear_idx = self.channels.index(self.segmentation_channel)
            nuclear_img = image[nuclear_idx]
            
            # Convert to uint8 for processing
            nuclear_uint8 = (nuclear_img * 255).astype(np.uint8)
            
            # Apply Gaussian filter to reduce noise
            smoothed = cv2.GaussianBlur(nuclear_uint8, (5, 5), 0)
            
            # Threshold using Otsu's method
            _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Fill holes
            binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
            
            # Watershed segmentation to separate touching nuclei
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            _, markers = cv2.connectedComponents(binary)
            
            # Apply watershed
            markers = cv2.watershed(cv2.cvtColor(nuclear_uint8, cv2.COLOR_GRAY2BGR), markers)
            
            # Create final segmentation mask
            segmentation_mask = (markers > 1).astype(np.uint8)
            
            # Extract cell features
            cell_features = self._extract_cell_features(image, markers)
            
            return {
                'mask': segmentation_mask,
                'features': cell_features,
                'num_cells': len(cell_features) if cell_features else 0
            }
            
        except Exception as e:
            print(f"WARNING: Segmentation error: {e}")
            return {'mask': None, 'features': None}
    
    def _extract_cell_features(self, image: np.ndarray, markers: np.ndarray) -> List[Dict]:
        """Extract features from segmented cells."""
        
        cell_features = []
        unique_labels = np.unique(markers)
        
        for label in unique_labels:
            if label <= 1:  # Skip background and watershed lines
                continue
            
            # Create cell mask
            cell_mask = (markers == label)
            
            # Skip if cell is too small or too large
            cell_area = np.sum(cell_mask)
            if cell_area < self.min_cell_area or cell_area > self.max_cell_area:
                continue
            
            # Extract morphological features
            props = measure.regionprops(cell_mask.astype(int))[0]
            
            # Basic morphology
            features = {
                'area': props.area,
                'perimeter': props.perimeter,
                'centroid': props.centroid,
                'eccentricity': props.eccentricity,
                'solidity': props.solidity,
                'extent': props.extent,
                'major_axis_length': props.major_axis_length,
                'minor_axis_length': props.minor_axis_length,
                'orientation': props.orientation,
            }
            
            # Intensity features for each channel
            for c, channel_name in enumerate(self.channels):
                channel_img = image[c]
                cell_intensities = channel_img[cell_mask]
                
                features[f'{channel_name}_mean'] = np.mean(cell_intensities)
                features[f'{channel_name}_std'] = np.std(cell_intensities)
                features[f'{channel_name}_median'] = np.median(cell_intensities)
                features[f'{channel_name}_max'] = np.max(cell_intensities)
                features[f'{channel_name}_min'] = np.min(cell_intensities)
                features[f'{channel_name}_total'] = np.sum(cell_intensities)
            
            # Texture features (simplified)
            nuclear_idx = self.channels.index(self.segmentation_channel) if self.segmentation_channel in self.channels else 0
            nuclear_intensities = image[nuclear_idx][cell_mask]
            
            # Compute texture measures
            features['texture_variance'] = np.var(nuclear_intensities)
            features['texture_entropy'] = self._compute_entropy(nuclear_intensities)
            
            cell_features.append(features)
        
        return cell_features
    
    def _compute_entropy(self, intensities: np.ndarray, bins: int = 32) -> float:
        """Compute entropy of intensity distribution."""
        
        try:
            hist, _ = np.histogram(intensities, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero entries
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return entropy
        except:
            return 0.0
    
    def compute_normalization_statistics(self, image_paths: List[Path]) -> Dict:
        """Compute global normalization statistics across dataset."""
        
        print("STATS: Computing global normalization statistics...")
        
        channel_stats = {channel: {'values': []} for channel in self.channels}
        
        # Sample images for statistics computation
        sample_size = min(1000, len(image_paths))
        sampled_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        for img_path in sampled_paths:
            try:
                # Load image (assuming multi-channel TIFF or similar)
                image = self._load_raw_image(img_path)
                
                for c, channel_name in enumerate(self.channels):
                    if c < image.shape[0]:
                        channel_img = image[c]
                        channel_stats[channel_name]['values'].extend(channel_img.flatten())
            
            except Exception as e:
                print(f"WARNING: Error processing {img_path}: {e}")
        
        # Compute percentiles for each channel
        for channel_name in self.channels:
            values = np.array(channel_stats[channel_name]['values'])
            
            if len(values) > 0:
                self.normalization_stats[channel_name] = {
                    'percentile_min': np.percentile(values, self.percentile_range[0]),
                    'percentile_max': np.percentile(values, self.percentile_range[1]),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            print(f"  {channel_name}: {self.normalization_stats[channel_name]}")
        
        return self.normalization_stats
    
    def _load_raw_image(self, image_path: Path) -> np.ndarray:
        """Load raw multi-channel image."""
        
        # This would depend on your image format
        # Example for TIFF files:
        try:
            import tifffile
            image = tifffile.imread(str(image_path))
            
            # Ensure correct dimensions [C, H, W]
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            elif image.ndim == 3 and image.shape[-1] <= 5:
                # Assume HWC format, convert to CHW
                image = np.transpose(image, (2, 0, 1))
            
            return image
            
        except ImportError:
            # Fallback to OpenCV for single channel
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            return image
    
    def batch_process_images(self, 
                           image_paths: List[Path],
                           metadata_list: List[Dict],
                           split: str = 'train',
                           batch_size: int = 32) -> List[Dict]:
        """Process multiple images in batches."""
        
        print(f"CYCLE: Batch processing {len(image_paths)} images...")
        
        processed_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]
            
            batch_results = []
            for img_path, metadata in zip(batch_paths, batch_metadata):
                try:
                    # Load image
                    image = self._load_raw_image(img_path)
                    
                    # Process image
                    result = self.process_image(image, metadata, split)
                    
                    if result is not None:
                        batch_results.append(result)
                
                except Exception as e:
                    print(f"WARNING: Error processing {img_path}: {e}")
            
            processed_results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_paths)}/{len(image_paths)} images")
        
        print(f"  SUCCESS: Successfully processed {len(processed_results)}/{len(image_paths)} images")
        return processed_results
    
    def create_image_montage(self, 
                           images: List[torch.Tensor],
                           titles: List[str] = None,
                           max_images: int = 16) -> np.ndarray:
        """Create a montage of processed images for visualization."""
        
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Limit number of images
        images = images[:max_images]
        titles = titles[:max_images] if titles else [f"Image {i+1}" for i in range(len(images))]
        
        # Calculate grid dimensions
        n_images = len(images)
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))
        
        # Create figure
        fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        for i, (image, title) in enumerate(zip(images, titles)):
            row = i // n_cols
            col = i % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Convert tensor to numpy if needed
            if isinstance(image, torch.Tensor):
                img_np = image.detach().cpu().numpy()
            else:
                img_np = image
            
            # Handle multi-channel images
            if img_np.ndim == 3:
                if img_np.shape[0] <= 3:  # CHW format
                    img_np = np.transpose(img_np[:3], (1, 2, 0))
                    if img_np.shape[2] == 1:
                        img_np = img_np.squeeze()
                        ax.imshow(img_np, cmap='gray')
                    else:
                        ax.imshow(img_np)
                else:  # More than 3 channels, show first channel
                    ax.imshow(img_np[0], cmap='gray')
            else:
                ax.imshow(img_np, cmap='gray')
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Remove empty subplots
        for i in range(n_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.add_subplot(gs[row, col]).axis('off')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        montage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        montage = montage.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return montage

class ImageAugmentationPipeline:
    """Advanced augmentation pipeline for cellular imaging."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Augmentation parameters
        self.augmentation_probability = config.get('augmentation_probability', 0.8)
        self.intensity_augmentation = config.get('intensity_augmentation', True)
        self.geometric_augmentation = config.get('geometric_augmentation', True)
        self.noise_augmentation = config.get('noise_augmentation', True)
        
        # Cellular-specific parameters
        self.preserve_cellular_structure = config.get('preserve_cellular_structure', True)
        self.max_rotation_angle = config.get('max_rotation_angle', 30)
        self.max_scale_factor = config.get('max_scale_factor', 0.2)
        
    def create_augmentation_pipeline(self, split: str = 'train') -> A.Compose:
        """Create augmentation pipeline based on split."""
        
        if split != 'train':
            # No augmentation for validation/test
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])
        
        transforms = []
        
        # Geometric augmentations
        if self.geometric_augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=self.max_scale_factor,
                    rotate_limit=self.max_rotation_angle,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.7
                ),
            ])
            
            # Elastic deformation (mild for cellular images)
            if not self.preserve_cellular_structure:
                transforms.append(
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        alpha_affine=10,
                        border_mode=cv2.BORDER_REFLECT,
                        p=0.3
                    )
                )
        
        # Intensity augmentations
        if self.intensity_augmentation:
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            ])
        
        # Noise augmentations
        if self.noise_augmentation:
            transforms.extend([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.2),
            ])
        
        # Blur augmentations (simulating focus issues)
        transforms.extend([
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
        ])
        
        # Final transformations
        transforms.extend([
            A.Resize(224, 224),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def apply_mixup(self, 
                   images: torch.Tensor, 
                   labels: torch.Tensor, 
                   alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation."""
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam
    
    def apply_cutmix(self, 
                    images: torch.Tensor, 
                    labels: torch.Tensor, 
                    alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation."""
        
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

class ImageQualityController:
    """Quality control system for cellular images."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Quality thresholds
        self.min_contrast = config.get('min_contrast', 0.1)
        self.max_saturation = config.get('max_saturation', 0.05)
        self.blur_threshold = config.get('blur_threshold', 100)
        self.artifact_threshold = config.get('artifact_threshold', 0.3)
        
        # Statistical parameters
        self.intensity_outlier_factor = config.get('intensity_outlier_factor', 3.0)
        
    def assess_image_quality(self, image: np.ndarray) -> Dict:
        """Comprehensive image quality assessment."""
        
        quality_metrics = {}
        
        # Convert to grayscale if multichannel
        if image.ndim == 3:
            if image.shape[0] <= 3:  # CHW format
                gray_image = np.mean(image, axis=0)
            else:
                gray_image = image[0]  # Use first channel
        else:
            gray_image = image
        
        # Contrast assessment
        quality_metrics['contrast'] = self._assess_contrast(gray_image)
        
        # Blur assessment
        quality_metrics['blur_score'] = self._assess_blur(gray_image)
        
        # Saturation assessment
        quality_metrics['saturation'] = self._assess_saturation(image)
        
        # Artifact detection
        quality_metrics['artifacts'] = self._detect_artifacts(gray_image)
        
        # Overall quality score
        quality_metrics['overall_score'] = self._compute_overall_quality(quality_metrics)
        
        # Pass/fail decision
        quality_metrics['passes_qc'] = self._passes_quality_control(quality_metrics)
        
        return quality_metrics
    
    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast using standard deviation."""
        return np.std(image) / (np.mean(image) + 1e-8)
    
    def _assess_blur(self, image: np.ndarray) -> float:
        """Assess image blur using Laplacian variance."""
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        return cv2.Laplacian(image_uint8, cv2.CV_64F).var()
    
    def _assess_saturation(self, image: np.ndarray) -> float:
        """Assess pixel saturation."""
        if image.ndim == 3:
            # Check each channel
            saturation_fractions = []
            for c in range(image.shape[0]):
                channel = image[c]
                max_val = np.max(channel)
                saturated_pixels = np.sum(channel >= max_val * 0.98)
                saturation_fractions.append(saturated_pixels / channel.size)
            return np.max(saturation_fractions)
        else:
            max_val = np.max(image)
            saturated_pixels = np.sum(image >= max_val * 0.98)
            return saturated_pixels / image.size
    
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detect imaging artifacts using edge detection."""
        
        # Compute gradients
        grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Look for unusually high gradients (potential artifacts)
        threshold = np.percentile(gradient_magnitude, 95)
        artifact_pixels = np.sum(gradient_magnitude > threshold * 2)
        
        return artifact_pixels / image.size
    
    def _compute_overall_quality(self, metrics: Dict) -> float:
        """Compute overall quality score (0-1, higher is better)."""
        
        # Normalize individual metrics
        contrast_score = min(1.0, metrics['contrast'] / 0.5)  # Good contrast > 0.5
        blur_score = min(1.0, metrics['blur_score'] / 200)   # Good blur score > 200
        saturation_score = 1.0 - min(1.0, metrics['saturation'] / 0.1)  # Low saturation is good
        artifact_score = 1.0 - min(1.0, metrics['artifacts'] / 0.2)  # Low artifacts is good
        
        # Weighted combination
        overall_score = (
            0.3 * contrast_score +
            0.3 * blur_score +
            0.2 * saturation_score +
            0.2 * artifact_score
        )
        
        return overall_score
    
    def _passes_quality_control(self, metrics: Dict) -> bool:
        """Determine if image passes quality control."""
        
        conditions = [
            metrics['contrast'] >= self.min_contrast,
            metrics['blur_score'] >= self.blur_threshold,
            metrics['saturation'] <= self.max_saturation,
            metrics['artifacts'] <= self.artifact_threshold,
            metrics['overall_score'] >= 0.5
        ]
        
        return all(conditions)
    
    def batch_quality_assessment(self, images: List[np.ndarray]) -> List[Dict]:
        """Assess quality for a batch of images."""
        
        quality_results = []
        
        for i, image in enumerate(images):
            try:
                quality_metrics = self.assess_image_quality(image)
                quality_results.append(quality_metrics)
            except Exception as e:
                print(f"WARNING: Error assessing quality for image {i}: {e}")
                quality_results.append({
                    'passes_qc': False,
                    'error': str(e)
                })
        
        return quality_results

class CellularFeatureExtractor:
    """Extract biological features from cellular images."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Feature extraction parameters
        self.extract_morphology = config.get('extract_morphology', True)
        self.extract_intensity = config.get('extract_intensity', True)
        self.extract_texture = config.get('extract_texture', True)
        self.extract_colocalization = config.get('extract_colocalization', True)
        
        # Channels for feature extraction
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5'])
        
    def extract_comprehensive_features(self, 
                                     image: np.ndarray,
                                     segmentation_mask: np.ndarray = None) -> Dict:
        """Extract comprehensive cellular features."""
        
        features = {}
        
        # Global image features
        features.update(self._extract_global_features(image))
        
        # Cell-level features (if segmentation available)
        if segmentation_mask is not None:
            cell_features = self._extract_cell_level_features(image, segmentation_mask)
            features.update(cell_features)
        
        # Channel-specific features
        for c, channel_name in enumerate(self.channels):
            if c < image.shape[0]:
                channel_features = self._extract_channel_features(
                    image[c], channel_name, segmentation_mask
                )
                features.update(channel_features)
        
        # Multi-channel features
        if image.shape[0] > 1:
            multi_channel_features = self._extract_multi_channel_features(image)
            features.update(multi_channel_features)
        
        return features
    
    def _extract_global_features(self, image: np.ndarray) -> Dict:
        """Extract global image-level features."""
        
        features = {}
        
        # Basic statistics for each channel
        for c, channel_name in enumerate(self.channels):
            if c < image.shape[0]:
                channel = image[c]
                
                features[f'global_{channel_name}_mean'] = np.mean(channel)
                features[f'global_{channel_name}_std'] = np.std(channel)
                features[f'global_{channel_name}_median'] = np.median(channel)
                features[f'global_{channel_name}_mad'] = np.median(np.abs(channel - np.median(channel)))
                features[f'global_{channel_name}_skewness'] = self._compute_skewness(channel)
                features[f'global_{channel_name}_kurtosis'] = self._compute_kurtosis(channel)
                
                # Percentiles
                percentiles = np.percentile(channel, [5, 25, 75, 95])
                features[f'global_{channel_name}_p5'] = percentiles[0]
                features[f'global_{channel_name}_p25'] = percentiles[1]
                features[f'global_{channel_name}_p75'] = percentiles[2]
                features[f'global_{channel_name}_p95'] = percentiles[3]
        
        return features
    
    def _extract_cell_level_features(self, 
                                   image: np.ndarray,
                                   segmentation_mask: np.ndarray) -> Dict:
        """Extract cell-level aggregated features."""
        
        features = {}
        
        # Get cell labels
        labels = measure.label(segmentation_mask)
        props = measure.regionprops(labels)
        
        if not props:
            return features
        
        # Aggregate morphological features
        cell_areas = [prop.area for prop in props]
        cell_perimeters = [prop.perimeter for prop in props]
        cell_eccentricities = [prop.eccentricity for prop in props]
        cell_solidities = [prop.solidity for prop in props]
        
        features['cell_count'] = len(props)
        features['cell_density'] = len(props) / (image.shape[1] * image.shape[2])
        
        # Area statistics
        features['cell_area_mean'] = np.mean(cell_areas)
        features['cell_area_std'] = np.std(cell_areas)
        features['cell_area_cv'] = np.std(cell_areas) / (np.mean(cell_areas) + 1e-8)
        
        # Shape statistics
        features['cell_eccentricity_mean'] = np.mean(cell_eccentricities)
        features['cell_solidity_mean'] = np.mean(cell_solidities)
        
        # Size distribution
        features['cell_area_range'] = np.max(cell_areas) - np.min(cell_areas)
        features['cell_size_uniformity'] = 1.0 / (1.0 + np.std(cell_areas) / np.mean(cell_areas))
        
        return features
    
    def _extract_channel_features(self, 
                                channel: np.ndarray,
                                channel_name: str,
                                segmentation_mask: np.ndarray = None) -> Dict:
        """Extract features for a specific channel."""
        
        features = {}
        prefix = f'channel_{channel_name}'
        
        # Intensity features
        if self.extract_intensity:
            features[f'{prefix}_total_intensity'] = np.sum(channel)
            features[f'{prefix}_mean_intensity'] = np.mean(channel)
            features[f'{prefix}_integrated_intensity'] = np.sum(channel * (channel > 0))
            
            # Intensity distribution
            hist, _ = np.histogram(channel, bins=32, density=True)
            features[f'{prefix}_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Texture features
        if self.extract_texture:
            texture_features = self._compute_texture_features(channel)
            for key, value in texture_features.items():
                features[f'{prefix}_{key}'] = value
        
        # Spatial features
        if segmentation_mask is not None:
            spatial_features = self._compute_spatial_features(channel, segmentation_mask)
            for key, value in spatial_features.items():
                features[f'{prefix}_{key}'] = value
        
        return features
    
    def _extract_multi_channel_features(self, image: np.ndarray) -> Dict:
        """Extract features that depend on multiple channels."""
        
        features = {}
        
        # Colocalization features
        if self.extract_colocalization and image.shape[0] >= 2:
            for i in range(image.shape[0]):
                for j in range(i + 1, image.shape[0]):
                    ch1_name = self.channels[i] if i < len(self.channels) else f'ch{i}'
                    ch2_name = self.channels[j] if j < len(self.channels) else f'ch{j}'
                    
                    colocalization = self._compute_colocalization(image[i], image[j])
                    features[f'colocalization_{ch1_name}_{ch2_name}'] = colocalization
        
        # Channel ratios
        if image.shape[0] >= 2:
            for i in range(image.shape[0]):
                for j in range(i + 1, image.shape[0]):
                    ch1_name = self.channels[i] if i < len(self.channels) else f'ch{i}'
                    ch2_name = self.channels[j] if j < len(self.channels) else f'ch{j}'
                    
                    ch1_total = np.sum(image[i])
                    ch2_total = np.sum(image[j])
                    ratio = ch1_total / (ch2_total + 1e-8)
                    features[f'ratio_{ch1_name}_{ch2_name}'] = ratio
        
        return features
    
    def _compute_texture_features(self, image: np.ndarray) -> Dict:
        """Compute texture features using GLCM and other methods."""
        
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert to uint8
        if image.dtype != np.uint8:
            image_uint8 = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        try:
            # Gray Level Co-occurrence Matrix
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(image_uint8, distances, angles, levels=32, symmetric=True, normed=True)
            
            # Texture properties
            texture_features = {}
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            
            for prop in props:
                values = graycoprops(glcm, prop)
                texture_features[f'texture_{prop}_mean'] = np.mean(values)
                texture_features[f'texture_{prop}_range'] = np.max(values) - np.min(values)
            
            # Additional texture measures
            texture_features['texture_local_variance'] = np.var(image)
            texture_features['texture_local_std'] = np.std(image)
            
            return texture_features
            
        except Exception as e:
            print(f"WARNING: Error computing texture features: {e}")
            return {}
    
    def _compute_spatial_features(self, 
                                channel: np.ndarray,
                                segmentation_mask: np.ndarray) -> Dict:
        """Compute spatial distribution features."""
        
        features = {}
        
        try:
            # Get cell regions
            labels = measure.label(segmentation_mask)
            props = measure.regionprops(labels, intensity_image=channel)
            
            if not props:
                return features
            
            # Spatial statistics
            centroids = np.array([prop.centroid for prop in props])
            
            if len(centroids) > 1:
                # Nearest neighbor distances
                from scipy.spatial.distance import pdist
                distances = pdist(centroids)
                
                features['spatial_nn_mean'] = np.mean(distances)
                features['spatial_nn_std'] = np.std(distances)
                features['spatial_nn_min'] = np.min(distances)
                
                # Spatial organization
                features['spatial_regularity'] = self._compute_spatial_regularity(centroids)
            
            # Intensity-based spatial features
            intensities = [prop.mean_intensity for prop in props]
            
            if len(intensities) > 1:
                features['spatial_intensity_variation'] = np.std(intensities) / (np.mean(intensities) + 1e-8)
                
                # Moran's I (spatial autocorrelation)
                features['spatial_morans_i'] = self._compute_morans_i(centroids, intensities)
        
        except Exception as e:
            print(f"WARNING: Error computing spatial features: {e}")
        
        return features
    
    def _compute_colocalization(self, channel1: np.ndarray, channel2: np.ndarray) -> float:
        """Compute colocalization coefficient between two channels."""
        
        # Flatten arrays
        ch1_flat = channel1.flatten()
        ch2_flat = channel2.flatten()
        
        # Remove zero pixels
        mask = (ch1_flat > 0) & (ch2_flat > 0)
        if np.sum(mask) == 0:
            return 0.0
        
        ch1_masked = ch1_flat[mask]
        ch2_masked = ch2_flat[mask]
        
        # Pearson correlation coefficient
        correlation = np.corrcoef(ch1_masked, ch2_masked)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _compute_spatial_regularity(self, centroids: np.ndarray) -> float:
        """Compute spatial regularity of cell positions."""
        
        if len(centroids) < 3:
            return 0.0
        
        from scipy.spatial.distance import pdist
        
        # Compute all pairwise distances
        distances = pdist(centroids)
        
        # Coefficient of variation of distances
        cv = np.std(distances) / (np.mean(distances) + 1e-8)
        
        # Regularity is inverse of coefficient of variation
        regularity = 1.0 / (1.0 + cv)
        
        return regularity
    
    def _compute_morans_i(self, positions: np.ndarray, values: np.ndarray) -> float:
        """Compute Moran's I spatial autocorrelation index."""
        
        if len(positions) < 3:
            return 0.0
        
        try:
            from scipy.spatial.distance import pdist, squareform
            
            # Compute distance matrix
            distances = squareform(pdist(positions))
            
            # Create weights matrix (inverse distance)
            weights = 1.0 / (distances + 1e-8)
            np.fill_diagonal(weights, 0)  # No self-weight
            
            # Row-normalize weights
            row_sums = np.sum(weights, axis=1)
            weights = weights / (row_sums[:, np.newaxis] + 1e-8)
            
            # Compute Moran's I
            n = len(values)
            mean_val = np.mean(values)
            deviations = values - mean_val
            
            numerator = np.sum(weights * np.outer(deviations, deviations))
            denominator = np.sum(deviations**2)
            
            morans_i = numerator / (denominator + 1e-8)
            
            return morans_i
            
        except Exception as e:
            print(f"WARNING: Error computing Moran's I: {e}")
            return 0.0
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
        return kurtosis

class MultiChannelNormalizer:
    """Advanced normalization for multi-channel cellular images."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Normalization methods
        self.method = config.get('method', 'percentile')  # percentile, zscore, minmax, robust
        self.percentile_range = config.get('percentile_range', (1, 99))
        self.robust_quantile_range = config.get('robust_quantile_range', (25, 75))
        
        # Channel-specific parameters
        self.channel_specific = config.get('channel_specific', True)
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5'])
        
        # Global statistics storage
        self.global_stats = {}
        
    def fit(self, images: List[np.ndarray]) -> None:
        """Fit normalizer on training data."""
        
        print("STATS: Computing normalization statistics...")
        
        # Initialize statistics storage
        for channel in self.channels:
            self.global_stats[channel] = {
                'values': [],
                'percentiles': {},
                'robust_stats': {},
                'zscore_stats': {}
            }
        
        # Collect statistics from sample of images
        sample_size = min(1000, len(images))
        sampled_images = np.random.choice(images, sample_size, replace=False)
        
        for img in sampled_images:
            if isinstance(img, str):
                # Load image if path is provided
                img = self._load_image(img)
            
            # Process each channel
            for c, channel in enumerate(self.channels):
                if c < img.shape[0]:
                    channel_data = img[c].flatten()
                    # Filter out extreme outliers
                    channel_data = channel_data[
                        (channel_data > np.percentile(channel_data, 0.1)) &
                        (channel_data < np.percentile(channel_data, 99.9))
                    ]
                    self.global_stats[channel]['values'].extend(channel_data)
        
        # Compute final statistics
        for channel in self.channels:
            if self.global_stats[channel]['values']:
                values = np.array(self.global_stats[channel]['values'])
                
                # Percentile statistics
                self.global_stats[channel]['percentiles'] = {
                    'p1': np.percentile(values, 1),
                    'p5': np.percentile(values, 5),
                    'p25': np.percentile(values, 25),
                    'p50': np.percentile(values, 50),
                    'p75': np.percentile(values, 75),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
                
                # Robust statistics
                q25, q75 = np.percentile(values, [25, 75])
                iqr = q75 - q25
                self.global_stats[channel]['robust_stats'] = {
                    'median': np.median(values),
                    'mad': np.median(np.abs(values - np.median(values))),
                    'q25': q25,
                    'q75': q75,
                    'iqr': iqr
                }
                
                # Z-score statistics
                self.global_stats[channel]['zscore_stats'] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                
                print(f"  {channel}: "
                      f"range=[{self.global_stats[channel]['percentiles']['p1']:.2f}, "
                      f"{self.global_stats[channel]['percentiles']['p99']:.2f}], "
                      f"median={self.global_stats[channel]['robust_stats']['median']:.2f}")
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform image using fitted statistics."""
        
        normalized_image = np.zeros_like(image, dtype=np.float32)
        
        for c, channel in enumerate(self.channels):
            if c < image.shape[0] and channel in self.global_stats:
                channel_data = image[c].astype(np.float32)
                
                if self.method == 'percentile':
                    vmin = self.global_stats[channel]['percentiles'][f'p{self.percentile_range[0]}']
                    vmax = self.global_stats[channel]['percentiles'][f'p{self.percentile_range[1]}']
                    normalized_image[c] = np.clip((channel_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
                
                elif self.method == 'zscore':
                    mean_val = self.global_stats[channel]['zscore_stats']['mean']
                    std_val = self.global_stats[channel]['zscore_stats']['std']
                    normalized_image[c] = (channel_data - mean_val) / (std_val + 1e-8)
                
                elif self.method == 'robust':
                    median_val = self.global_stats[channel]['robust_stats']['median']
                    mad_val = self.global_stats[channel]['robust_stats']['mad']
                    normalized_image[c] = (channel_data - median_val) / (mad_val + 1e-8)
                
                elif self.method == 'minmax':
                    vmin = self.global_stats[channel]['percentiles']['p1']
                    vmax = self.global_stats[channel]['percentiles']['p99']
                    normalized_image[c] = (channel_data - vmin) / (vmax - vmin + 1e-8)
                
                else:
                    raise ValueError(f"Unknown normalization method: {self.method}")
            else:
                # Fallback to per-image normalization
                channel_data = image[c].astype(np.float32)
                vmin, vmax = np.percentile(channel_data, [1, 99])
                normalized_image[c] = np.clip((channel_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        return normalized_image
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        # Implementation depends on image format
        import tifffile
        return tifffile.imread(image_path)

class ImageDataValidator:
    """Validator for image data integrity and format compliance."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Validation parameters
        self.expected_channels = config.get('expected_channels', 5)
        self.expected_shape = config.get('expected_shape', None)
        self.allowed_dtypes = config.get('allowed_dtypes', [np.uint8, np.uint16, np.float32])
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        
        # Quality thresholds
        self.min_dynamic_range = config.get('min_dynamic_range', 10)
        self.max_zero_fraction = config.get('max_zero_fraction', 0.9)
        
    def validate_image_batch(self, image_paths: List[Path]) -> Dict:
        """Validate a batch of images."""
        
        print(f"SEARCH: Validating {len(image_paths)} images...")
        
        validation_results = {
            'total_images': len(image_paths),
            'valid_images': 0,
            'invalid_images': 0,
            'errors': {},
            'statistics': {}
        }
        
        error_counts = {}
        
        for i, img_path in enumerate(image_paths):
            try:
                validation_result = self.validate_single_image(img_path)
                
                if validation_result['valid']:
                    validation_results['valid_images'] += 1
                else:
                    validation_results['invalid_images'] += 1
                    
                    # Count error types
                    for error in validation_result['errors']:
                        error_counts[error] = error_counts.get(error, 0) + 1
                
                # Progress reporting
                if (i + 1) % 100 == 0:
                    print(f"  Validated {i + 1}/{len(image_paths)} images")
            
            except Exception as e:
                validation_results['invalid_images'] += 1
                error_counts[f'processing_error'] = error_counts.get('processing_error', 0) + 1
                print(f"WARNING: Error validating {img_path}: {e}")
        
        validation_results['errors'] = error_counts
        
        # Compute statistics
        valid_fraction = validation_results['valid_images'] / validation_results['total_images']
        validation_results['statistics'] = {
            'valid_fraction': valid_fraction,
            'most_common_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        print(f"  SUCCESS: Validation complete: {validation_results['valid_images']}/{validation_results['total_images']} "
              f"({valid_fraction:.1%}) images passed validation")
        
        return validation_results
    
    def validate_single_image(self, image_path: Path) -> Dict:
        """Validate a single image file."""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Check file existence and size
            if not image_path.exists():
                validation_result['errors'].append('file_not_found')
                validation_result['valid'] = False
                return validation_result
            
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                validation_result['errors'].append('file_too_large')
                validation_result['valid'] = False
            
            # Load and check image
            try:
                import tifffile
                image = tifffile.imread(str(image_path))
            except Exception as e:
                validation_result['errors'].append('load_error')
                validation_result['valid'] = False
                return validation_result
            
            # Check image properties
            validation_result['metadata'] = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'file_size_mb': file_size_mb
            }
            
            # Validate dimensions
            if image.ndim not in [2, 3]:
                validation_result['errors'].append('invalid_dimensions')
                validation_result['valid'] = False
            
            # Check number of channels for multi-channel images
            if image.ndim == 3 and self.expected_channels:
                if image.shape[0] != self.expected_channels:
                    validation_result['errors'].append('channel_mismatch')
                    validation_result['valid'] = False
            
            # Check data type
            if image.dtype not in self.allowed_dtypes:
                validation_result['errors'].append('invalid_dtype')
                validation_result['valid'] = False
            
            # Check dynamic range
            if image.ndim == 2:
                dynamic_range = np.max(image) - np.min(image)
            else:
                dynamic_range = np.mean([np.max(image[c]) - np.min(image[c]) for c in range(image.shape[0])])
            
            if dynamic_range < self.min_dynamic_range:
                validation_result['errors'].append('low_dynamic_range')
                validation_result['valid'] = False
            
            # Check for excessive zeros
            zero_fraction = np.mean(image == 0)
            if zero_fraction > self.max_zero_fraction:
                validation_result['errors'].append('excessive_zeros')
                validation_result['valid'] = False
            
            validation_result['metadata'].update({
                'dynamic_range': float(dynamic_range),
                'zero_fraction': float(zero_fraction)
            })
            
        except Exception as e:
            validation_result['errors'].append('processing_error')
            validation_result['valid'] = False
        
        return validation_result