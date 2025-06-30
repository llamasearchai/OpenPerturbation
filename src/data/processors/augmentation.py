"""
Advanced data augmentation for cellular imaging.

Includes biology-aware augmentations that preserve cellular structure
while providing effective data augmentation for deep learning.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from scipy import ndimage
from skimage import morphology, filters, exposure
import warnings


class BiologyAwareAugmentations:
    """Biology-aware augmentations that preserve cellular structure."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.channels = config.get("channels", ["DAPI", "GFP", "RFP", "Cy5", "Cy7"])
        self.nuclear_channel = config.get("nuclear_channel", "DAPI")
        self.preserve_nuclear_structure = config.get("preserve_nuclear_structure", True)

        # Augmentation probabilities
        self.geometric_prob = config.get("geometric_prob", 0.8)
        self.intensity_prob = config.get("intensity_prob", 0.7)
        self.noise_prob = config.get("noise_prob", 0.5)
        self.blur_prob = config.get("blur_prob", 0.3)
        self.elastic_prob = config.get("elastic_prob", 0.4)

        # Intensity augmentation parameters
        self.brightness_range = config.get("brightness_range", 0.2)
        self.contrast_range = config.get("contrast_range", 0.2)
        self.gamma_range = config.get("gamma_range", [0.8, 1.2])

        # Noise parameters
        self.noise_std_range = config.get("noise_std_range", [0.01, 0.05])
        self.poisson_noise_lambda = config.get("poisson_noise_lambda", 0.1)

        # Microscopy-specific parameters
        self.simulate_bleaching = config.get("simulate_bleaching", True)
        self.simulate_illumination = config.get("simulate_illumination", True)
        self.simulate_defocus = config.get("simulate_defocus", True)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        perturbation_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Apply biology-aware augmentations.

        Args:
            image: Multi-channel image [C, H, W]
            mask: Segmentation mask [H, W] (optional)
            perturbation_info: Information about perturbation (optional)

        Returns:
            Dictionary with augmented image and mask
        """

        # Ensure image is float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Apply augmentations
        augmented = {
            "image": image.copy(),
            "mask": mask.copy() if mask is not None else None,
            "perturbation_info": perturbation_info,
        }

        # Geometric augmentations
        if random.random() < self.geometric_prob:
            augmented = self._apply_geometric_augmentations(augmented)

        # Intensity augmentations (channel-specific)
        if random.random() < self.intensity_prob:
            augmented = self._apply_intensity_augmentations(augmented)

        # Microscopy-specific augmentations
        if self.simulate_illumination and random.random() < 0.4:
            augmented = self._simulate_uneven_illumination(augmented)

        if self.simulate_bleaching and random.random() < 0.3:
            augmented = self._simulate_photobleaching(augmented)

        if self.simulate_defocus and random.random() < 0.2:
            augmented = self._simulate_defocus_blur(augmented)

        # Noise augmentations
        if random.random() < self.noise_prob:
            augmented = self._apply_noise_augmentations(augmented)

        # Elastic deformations (careful with cellular structures)
        if random.random() < self.elastic_prob:
            augmented = self._apply_elastic_deformation(augmented)

        return augmented

    def _apply_geometric_augmentations(self, data: Dict) -> Dict:
        """Apply geometric augmentations."""

        image = data["image"]
        mask = data["mask"]

        # Define geometric transforms
        geometric_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.7, border_mode=cv2.BORDER_REFLECT),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.6,
            ),
            A.Transpose(p=0.3),
        ]

        # Apply transforms
        transform = A.Compose(geometric_transforms)

        # Prepare data for albumentations
        if image.ndim == 3:
            # Multi-channel image: process each channel separately to maintain relationships
            augmented_channels = []

            for c in range(image.shape[0]):
                channel_data = {
                    "image": image[c],
                    "mask": mask if mask is not None else np.zeros_like(image[c]),
                }

                augmented_channel = transform(**channel_data)
                augmented_channels.append(augmented_channel["image"])

                # Update mask only once (same transform for all channels)
                if c == 0 and mask is not None:
                    mask = augmented_channel["mask"]

            image = np.stack(augmented_channels, axis=0)
        else:
            # Single channel
            transform_data = {
                "image": image,
                "mask": mask if mask is not None else np.zeros_like(image),
            }

            augmented = transform(**transform_data)
            image = augmented["image"]
            if mask is not None:
                mask = augmented["mask"]

        data["image"] = image
        data["mask"] = mask

        return data

    def _apply_intensity_augmentations(self, data: Dict) -> Dict:
        """Apply channel-specific intensity augmentations."""

        image = data["image"]

        # Apply augmentations per channel
        for c, channel_name in enumerate(self.channels):
            if c >= image.shape[0]:
                continue

            channel = image[c]

            # Brightness and contrast adjustment
            if random.random() < 0.7:
                brightness_factor = 1.0 + random.uniform(
                    -self.brightness_range, self.brightness_range
                )
                contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)

                channel = channel * contrast_factor + (brightness_factor - 1.0) * np.mean(channel)

            # Gamma correction
            if random.random() < 0.5:
                gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
                channel = self._apply_gamma_correction(channel, gamma)

            # Channel-specific augmentations
            if channel_name == self.nuclear_channel and self.preserve_nuclear_structure:
                # More conservative augmentations for nuclear channel
                channel = self._apply_conservative_intensity_aug(channel)
            else:
                # Standard intensity augmentations for other channels
                channel = self._apply_standard_intensity_aug(channel)

            # Histogram equalization (occasionally)
            if random.random() < 0.2:
                channel = self._apply_adaptive_histogram_equalization(channel)

            image[c] = channel

        data["image"] = image

        return data

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction."""

        # Normalize to [0, 1]
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Apply gamma correction
        corrected = np.power(image_norm, gamma)

        # Rescale to original range
        corrected = corrected * (image.max() - image.min()) + image.min()

        return corrected

    def _apply_conservative_intensity_aug(self, channel: np.ndarray) -> np.ndarray:
        """Apply conservative intensity augmentations for nuclear channel."""

        # Mild intensity variations
        if random.random() < 0.5:
            noise_std = random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_std, channel.shape)
            channel = channel + noise * np.std(channel)

        # Mild contrast adjustment
        if random.random() < 0.4:
            contrast_factor = random.uniform(0.9, 1.1)
            mean_intensity = np.mean(channel)
            channel = (channel - mean_intensity) * contrast_factor + mean_intensity

        return channel

    def _apply_standard_intensity_aug(self, channel: np.ndarray) -> np.ndarray:
        """Apply standard intensity augmentations."""

        # Intensity scaling
        if random.random() < 0.6:
            scale_factor = random.uniform(0.8, 1.2)
            channel = channel * scale_factor

        # Intensity offset
        if random.random() < 0.4:
            offset = random.uniform(-0.1, 0.1) * np.mean(channel)
            channel = channel + offset

        return channel

    def _apply_adaptive_histogram_equalization(self, channel: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization."""

        try:
            # Convert to uint16 for CLAHE
            channel_uint16 = (
                (channel - channel.min()) / (channel.max() - channel.min()) * 65535
            ).astype(np.uint16)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(channel_uint16)

            # Convert back to float32
            equalized = equalized.astype(np.float32) / 65535.0
            equalized = equalized * (channel.max() - channel.min()) + channel.min()

            return equalized

        except Exception:
            return channel

    def _simulate_uneven_illumination(self, data: Dict) -> Dict:
        """Simulate uneven illumination field."""

        image = data["image"]
        h, w = image.shape[-2:]

        # Create illumination field
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2

        # Gaussian illumination pattern
        sigma_y = random.uniform(h * 0.3, h * 0.7)
        sigma_x = random.uniform(w * 0.3, w * 0.7)

        # Create Gaussian illumination field
        illumination = np.exp(
            -((x - center_x) ** 2 / (2 * sigma_x**2) + (y - center_y) ** 2 / (2 * sigma_y**2))
        )

        # Add some randomness
        illumination = illumination * random.uniform(0.7, 1.3)
        illumination = illumination + random.uniform(0.3, 0.7)

        # Smooth the illumination field
        illumination = cv2.GaussianBlur(illumination.astype(np.float32), (21, 21), 0)

        # Apply to all channels
        for c in range(image.shape[0]):
            image[c] = image[c] * illumination

        data["image"] = image

        return data

    def _simulate_photobleaching(self, data: Dict) -> Dict:
        """Simulate photobleaching effects."""

        image = data["image"]

        # Simulate bleaching with exponential decay
        for c in range(image.shape[0]):
            if random.random() < 0.5:  # Only bleach some channels
                # Exponential decay parameters
                decay_rate = random.uniform(0.05, 0.2)

                # Create spatial decay pattern
                h, w = image.shape[-2:]
                decay_pattern = np.ones((h, w))

                # Add some spatial variation to bleaching
                for _ in range(random.randint(1, 3)):
                    center_y = random.randint(h // 4, 3 * h // 4)
                    center_x = random.randint(w // 4, 3 * w // 4)
                    radius = random.randint(min(h, w) // 8, min(h, w) // 4)

                    y, x = np.ogrid[:h, :w]
                    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
                    decay_pattern[mask] *= 1 - decay_rate

                image[c] = image[c] * decay_pattern

        data["image"] = image

        return data

    def _simulate_defocus_blur(self, data: Dict) -> Dict:
        """Simulate defocus blur."""

        image = data["image"]

        # Apply slight defocus blur
        blur_sigma = random.uniform(0.5, 2.0)

        for c in range(image.shape[0]):
            if random.random() < 0.6:  # Not all channels need blur
                image[c] = cv2.GaussianBlur(image[c], (0, 0), blur_sigma)

        data["image"] = image

        return data

    def _apply_noise_augmentations(self, data: Dict) -> Dict:
        """Apply various noise augmentations."""

        image = data["image"]

        for c in range(image.shape[0]):
            channel = image[c]

            # Gaussian noise
            if random.random() < 0.6:
                noise_std = random.uniform(self.noise_std_range[0], self.noise_std_range[1])
                noise = np.random.normal(0, noise_std * np.std(channel), channel.shape)
                channel = channel + noise

            # Poisson noise
            if random.random() < 0.3:
                # Scale to positive values for Poisson
                channel_positive = channel - channel.min() + 1e-6
                poisson_noise = np.random.poisson(channel_positive * self.poisson_noise_lambda)
                channel = channel + (poisson_noise - channel_positive * self.poisson_noise_lambda)

            # Salt and pepper noise
            if random.random() < 0.2:
                salt_pepper_prob = random.uniform(0.001, 0.01)
                noise_mask = np.random.random(channel.shape) < salt_pepper_prob
                salt_mask = np.random.random(channel.shape) < 0.5

                channel[noise_mask & salt_mask] = channel.max()
                channel[noise_mask & ~salt_mask] = channel.min()

            image[c] = channel

        data["image"] = image

        return data

    def _apply_elastic_deformation(self, data: Dict) -> Dict:
        """Apply elastic deformation while preserving cellular structures."""

        image = data["image"]
        mask = data["mask"]

        # Conservative elastic deformation parameters
        alpha = random.uniform(50, 150)  # Deformation strength
        sigma = random.uniform(8, 12)  # Smoothing sigma

        # Generate random displacement fields
        h, w = image.shape[-2:]
        dx = np.random.uniform(-1, 1, (h, w)) * alpha
        dy = np.random.uniform(-1, 1, (h, w)) * alpha

        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = (x + dx).astype(np.float32)
        y_new = (y + dy).astype(np.float32)

        # Apply deformation to all channels
        for c in range(image.shape[0]):
            image[c] = cv2.remap(
                image[c], x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

        # Apply deformation to mask
        if mask is not None:
            mask = cv2.remap(
                mask.astype(np.float32),
                x_new,
                y_new,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT,
            )
            mask = mask.astype(bool)

        data["image"] = image
        data["mask"] = mask

        return data


class PerturbationAwareAugmentations:
    """Augmentations that are aware of perturbation context."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.perturbation_types = config.get(
            "perturbation_types", ["control", "knockout", "overexpression", "drug_treatment"]
        )

        # Perturbation-specific augmentation strengths
        self.augmentation_strengths = {
            "control": 1.0,
            "knockout": 0.8,  # More conservative for knockout
            "overexpression": 1.2,  # Stronger for overexpression
            "drug_treatment": 1.1,
        }

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        perturbation_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Apply perturbation-aware augmentations.

        Args:
            image: Multi-channel image [C, H, W]
            mask: Segmentation mask [H, W] (optional)
            perturbation_info: Information about perturbation

        Returns:
            Dictionary with augmented image and mask
        """

        # Get perturbation type
        perturbation_type = "control"
        if perturbation_info is not None:
            perturbation_type = perturbation_info.get("type", "control")

        # Get augmentation strength for this perturbation type
        strength = self.augmentation_strengths.get(perturbation_type, 1.0)

        # Apply perturbation-specific augmentations
        augmented = {
            "image": image.copy(),
            "mask": mask.copy() if mask is not None else None,
            "perturbation_info": perturbation_info,
        }

        # Morphological augmentations based on perturbation type
        if perturbation_type == "knockout":
            augmented = self._apply_knockout_augmentations(augmented, strength)
        elif perturbation_type == "overexpression":
            augmented = self._apply_overexpression_augmentations(augmented, strength)
        elif perturbation_type == "drug_treatment":
            augmented = self._apply_drug_treatment_augmentations(augmented, strength)

        return augmented

    def _apply_knockout_augmentations(self, data: Dict, strength: float) -> Dict:
        """Apply augmentations simulating knockout effects."""

        image = data["image"]

        # Simulate reduced protein expression
        target_channels = [1, 2, 3]  # Non-nuclear channels

        for c in target_channels:
            if c < image.shape[0] and random.random() < 0.4:
                # Reduce intensity to simulate knockout
                reduction_factor = random.uniform(0.5, 0.8) * strength
                image[c] = image[c] * reduction_factor

                # Add some spatial heterogeneity
                if random.random() < 0.3:
                    h, w = image.shape[-2:]
                    noise = np.random.uniform(0.8, 1.2, (h, w))
                    noise = cv2.GaussianBlur(noise, (11, 11), 0)
                    image[c] = image[c] * noise

        data["image"] = image

        return data

    def _apply_overexpression_augmentations(self, data: Dict, strength: float) -> Dict:
        """Apply augmentations simulating overexpression effects."""

        image = data["image"]

        # Simulate increased protein expression
        target_channels = [1, 2, 3]  # Non-nuclear channels

        for c in target_channels:
            if c < image.shape[0] and random.random() < 0.4:
                # Increase intensity to simulate overexpression
                amplification_factor = random.uniform(1.2, 1.8) * strength
                image[c] = image[c] * amplification_factor

                # Add punctate structures to simulate protein aggregation
                if random.random() < 0.3:
                    image[c] = self._add_punctate_structures(image[c])

        data["image"] = image

        return data

    def _apply_drug_treatment_augmentations(self, data: Dict, strength: float) -> Dict:
        """Apply augmentations simulating drug treatment effects."""

        image = data["image"]

        # Simulate various drug effects
        if random.random() < 0.3:
            # Cytotoxic effects - cell shrinkage simulation
            image = self._simulate_cell_shrinkage(image, strength)

        if random.random() < 0.2:
            # Membrane effects - boundary changes
            image = self._simulate_membrane_effects(image, strength)

        if random.random() < 0.4:
            # Metabolic effects - intensity changes
            image = self._simulate_metabolic_effects(image, strength)

        data["image"] = image

        return data

    def _add_punctate_structures(self, channel: np.ndarray) -> np.ndarray:
        """Add punctate structures to simulate protein aggregation."""

        h, w = channel.shape
        num_puncta = random.randint(5, 20)

        for _ in range(num_puncta):
            # Random location
            y = random.randint(5, h - 5)
            x = random.randint(5, w - 5)

            # Random size
            radius = random.randint(1, 3)

            # Add bright spot
            intensity = channel.max() * random.uniform(1.5, 2.0)

            yy, xx = np.ogrid[:h, :w]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            channel[mask] = np.maximum(channel[mask], intensity)

        return channel

    def _simulate_cell_shrinkage(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Simulate cell shrinkage effects."""

        # Apply mild morphological erosion to simulate shrinkage
        kernel_size = int(3 * strength)
        if kernel_size > 1:
            kernel = morphology.disk(kernel_size)

            for c in range(image.shape[0]):
                if c > 0:  # Skip nuclear channel
                    eroded = morphology.erosion(image[c], kernel)
                    image[c] = 0.7 * image[c] + 0.3 * eroded

        return image

    def _simulate_membrane_effects(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Simulate membrane permeabilization effects."""

        # Add edge enhancement to simulate membrane changes
        for c in range(1, image.shape[0]):  # Skip nuclear channel
            if random.random() < 0.5:
                # Edge detection
                edges = cv2.Canny(image[c].astype(np.uint8), 30, 100)
                edges = edges.astype(np.float32) / 255.0

                # Add edge enhancement
                image[c] = image[c] + edges * strength * 0.2

        return image

    def _simulate_metabolic_effects(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Simulate metabolic changes."""

        # Random intensity modulation to simulate metabolic changes
        for c in range(image.shape[0]):
            if random.random() < 0.6:
                modulation = random.uniform(0.8, 1.2) * strength
                image[c] = image[c] * modulation

        return image


class MixupAugmentation:
    """Mixup augmentation for perturbation biology."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self, batch_images: torch.Tensor, batch_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation to a batch.

        Args:
            batch_images: Batch of images [B, C, H, W]
            batch_labels: Batch of labels [B, ...]

        Returns:
            Mixed images, original labels, shuffled labels, lambda
        """
        batch_size = batch_images.size(0)

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        # Random permutation
        index = torch.randperm(batch_size).to(batch_images.device)

        # Mix images
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index, :]

        # Return mixed images and labels for loss computation
        return mixed_images, batch_labels, batch_labels[index], lam


class CutMixAugmentation:
    """CutMix augmentation adapted for cellular imaging."""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self, batch_images: torch.Tensor, batch_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation to a batch.

        Args:
            batch_images: Batch of images [B, C, H, W]
            batch_labels: Batch of labels [B, ...]

        Returns:
            Mixed images, original labels, shuffled labels, lambda
        """

        if np.random.rand() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0

        batch_size = batch_images.size(0)

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation
        index = torch.randperm(batch_size).to(batch_images.device)

        # Generate random bounding box
        W = batch_images.size(3)
        H = batch_images.size(2)

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling of bounding box center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        batch_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to reflect actual area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_images.size(-1) * batch_images.size(-2)))

        return batch_images, batch_labels, batch_labels[index], lam


class PerturbationDataAugmentation:
    """Comprehensive augmentation pipeline for perturbation biology."""

    def __init__(self, config: DictConfig, mode: str = "train"):
        self.config = config
        self.mode = mode

        # Initialize augmentation components
        self.biology_aware = BiologyAwareAugmentations(config.biology_aware)
        self.perturbation_aware = PerturbationAwareAugmentations(config.perturbation_aware)

        # Advanced augmentations
        self.use_mixup = config.get("use_mixup", False)
        self.use_cutmix = config.get("use_cutmix", False)

        if self.use_mixup:
            self.mixup = MixupAugmentation(alpha=config.mixup.alpha)

        if self.use_cutmix:
            self.cutmix = CutMixAugmentation(alpha=config.cutmix.alpha, prob=config.cutmix.prob)

        # Normalization parameters (per channel)
        self.mean = config.get("normalization_mean", [0.5] * 5)
        self.std = config.get("normalization_std", [0.5] * 5)

        # Build albumentations pipeline
        self.albumentations_pipeline = self._build_albumentations_pipeline()

    def _build_albumentations_pipeline(self) -> A.Compose:
        """Build albumentations pipeline for basic preprocessing."""

        transforms = []

        if self.mode == "train":
            # Training augmentations
            transforms.extend(
                [
                    A.RandomResizedCrop(
                        height=self.config.image_size,
                        width=self.config.image_size,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                        ],
                        p=0.2,
                    ),
                ]
            )
        else:
            # Validation/test preprocessing
            transforms.append(A.Resize(height=self.config.image_size, width=self.config.image_size))

        # Common transforms
        transforms.extend(
            [
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

        return A.Compose(transforms)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        perturbation_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Apply complete augmentation pipeline.

        Args:
            image: Multi-channel image [C, H, W]
            mask: Segmentation mask [H, W] (optional)
            perturbation_info: Perturbation information (optional)

        Returns:
            Dictionary with augmented data
        """

        result = {"image": image, "mask": mask, "perturbation_info": perturbation_info}

        if self.mode == "train":
            # Apply biology-aware augmentations
            result = self.biology_aware(
                image=result["image"],
                mask=result["mask"],
                perturbation_info=result["perturbation_info"],
            )

            # Apply perturbation-aware augmentations
            result = self.perturbation_aware(
                image=result["image"],
                mask=result["mask"],
                perturbation_info=result["perturbation_info"],
            )

        # Apply albumentations pipeline
        # Convert multi-channel image to format expected by albumentations
        if result["image"].ndim == 3:
            # Take first 3 channels or pad to 3 channels for albumentations
            if result["image"].shape[0] >= 3:
                albu_image = np.transpose(result["image"][:3], (1, 2, 0))
            else:
                # Pad channels
                padded_image = np.zeros((3, result["image"].shape[1], result["image"].shape[2]))
                padded_image[: result["image"].shape[0]] = result["image"]
                albu_image = np.transpose(padded_image, (1, 2, 0))
        else:
            # Single channel - convert to 3 channel
            albu_image = np.stack([result["image"]] * 3, axis=-1)

        # Prepare data for albumentations
        albu_data = {"image": albu_image}
        if result["mask"] is not None:
            albu_data["mask"] = result["mask"]

        # Apply albumentations
        augmented = self.albumentations_pipeline(**albu_data)

        # Convert back to original format
        augmented_image = augmented["image"]  # Now a tensor

        # If we had more than 3 channels originally, we need to handle them
        if result["image"].shape[0] > 3:
            # Apply same transform to remaining channels
            remaining_channels = []
            for c in range(3, result["image"].shape[0]):
                channel_data = {"image": result["image"][c]}
                # Apply geometric transforms only (no normalization)
                geom_pipeline = self._build_geometric_pipeline()
                transformed_channel = geom_pipeline(**channel_data)
                remaining_channels.append(torch.tensor(transformed_channel["image"]).unsqueeze(0))

            if remaining_channels:
                remaining_tensor = torch.cat(remaining_channels, dim=0)
                augmented_image = torch.cat([augmented_image, remaining_tensor], dim=0)

        result["image"] = augmented_image
        if "mask" in augmented:
            result["mask"] = augmented["mask"]

        return result

    def _build_geometric_pipeline(self) -> A.Compose:
        """Build pipeline with only geometric transforms (no normalization)."""

        transforms = []

        if self.mode == "train":
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=90, p=0.7),
                    A.Resize(height=self.config.image_size, width=self.config.image_size),
                ]
            )
        else:
            transforms.append(A.Resize(height=self.config.image_size, width=self.config.image_size))

        return A.Compose(transforms)

    def apply_batch_augmentations(
        self, batch_images: torch.Tensor, batch_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply batch-level augmentations like MixUp and CutMix.

        Args:
            batch_images: Batch of images [B, C, H, W]
            batch_labels: Batch of labels [B, ...]

        Returns:
            Augmented batch data
        """

        if self.mode != "train":
            return batch_images, batch_labels, batch_labels, 1.0

        # Randomly choose between different batch augmentations
        if self.use_mixup and self.use_cutmix:
            if random.random() < 0.5:
                return self.mixup(batch_images, batch_labels)
            else:
                return self.cutmix(batch_images, batch_labels)
        elif self.use_mixup:
            return self.mixup(batch_images, batch_labels)
        elif self.use_cutmix:
            return self.cutmix(batch_images, batch_labels)
        else:
            return batch_images, batch_labels, batch_labels, 1.0


def create_augmentation_pipeline(
    config: DictConfig, mode: str = "train"
) -> PerturbationDataAugmentation:
    """Factory function to create augmentation pipeline."""

    return PerturbationDataAugmentation(config, mode)


def test_augmentations():
    """Test augmentation pipeline."""

    print("TEST: Testing Augmentation Pipeline...")

    # Create synthetic test data
    np.random.seed(42)
    test_image = np.random.rand(5, 256, 256).astype(np.float32)
    test_mask = np.random.rand(256, 256) > 0.5

    # Test configuration
    config = DictConfig(
        {
            "biology_aware": {
                "channels": ["DAPI", "GFP", "RFP", "Cy5", "Cy7"],
                "nuclear_channel": "DAPI",
                "geometric_prob": 0.8,
                "intensity_prob": 0.7,
                "noise_prob": 0.5,
                "simulate_bleaching": True,
                "simulate_illumination": True,
                "simulate_defocus": True,
            },
            "perturbation_aware": {
                "perturbation_types": ["control", "knockout", "overexpression", "drug_treatment"],
            },
            "image_size": 224,
            "normalization_mean": [0.5] * 5,
            "normalization_std": [0.5] * 5,
            "use_mixup": True,
            "use_cutmix": True,
            "mixup": {"alpha": 0.2},
            "cutmix": {"alpha": 1.0, "prob": 0.5},
        }
    )

    # Create augmentation pipeline
    aug_pipeline = create_augmentation_pipeline(config, mode="train")

    # Test single image augmentation
    print("  CAMERA: Testing single image augmentation...")
    perturbation_info = {"type": "knockout", "target": "TP53"}

    result = aug_pipeline(image=test_image, mask=test_mask, perturbation_info=perturbation_info)

    print(f"    SUCCESS: Original image shape: {test_image.shape}")
    print(f"    SUCCESS: Augmented image shape: {result['image'].shape}")
    print(f"    SUCCESS: Augmented image type: {type(result['image'])}")

    # Test batch augmentation
    print("  CYCLE: Testing batch augmentation...")
    batch_images = torch.randn(8, 5, 224, 224)
    batch_labels = torch.randint(0, 10, (8,))

    mixed_images, labels_a, labels_b, lam = aug_pipeline.apply_batch_augmentations(
        batch_images, batch_labels
    )

    print(f"    SUCCESS: Batch mixed images shape: {mixed_images.shape}")
    print(f"    SUCCESS: Lambda value: {lam:.3f}")

    # Test different perturbation types
    print("   Testing perturbation-specific augmentations...")
    perturbation_types = ["control", "knockout", "overexpression", "drug_treatment"]

    for ptype in perturbation_types:
        perturbation_info = {"type": ptype, "target": "test_gene"}
        result = aug_pipeline(image=test_image, mask=test_mask, perturbation_info=perturbation_info)
        print(f"    SUCCESS: {ptype}: Image augmented successfully")

    print("COMPLETE: Augmentation pipeline test completed successfully!")


if __name__ == "__main__":
    test_augmentations()
