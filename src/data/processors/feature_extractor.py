"""
Advanced feature extraction pipeline for cellular imaging data.

This module provides comprehensive feature extraction capabilities including:
- Morphological features (shape, size, texture)
- Intensity features (distribution, spatial patterns)
- Spatial features (organization, clustering)
- Multi-scale features (local and global patterns)
- Biological features (organelle detection, phenotype classification)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import cv2
from scipy import ndimage
from skimage import measure, feature, filters, segmentation, morphology
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

class ComprehensiveFeatureExtractor:
    """
    Comprehensive feature extraction pipeline for cellular imaging.
    
    Extracts multiple types of features:
    - Morphological: cell shape, size, texture
    - Intensity: distribution statistics, spatial patterns
    - Spatial: organization, clustering, nearest neighbors
    - Multi-scale: local and global patterns
    - Biological: organelle detection, phenotype markers
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Feature extraction configuration
        self.extract_morphology = config.get('extract_morphology', True)
        self.extract_intensity = config.get('extract_intensity', True)
        self.extract_texture = config.get('extract_texture', True)
        self.extract_spatial = config.get('extract_spatial', True)
        self.extract_colocalization = config.get('extract_colocalization', True)
        self.extract_organelles = config.get('extract_organelles', True)
        
        # Channel configuration
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5', 'Cy7'])
        self.nuclear_channel = config.get('nuclear_channel', 'DAPI')
        self.cytoplasm_channels = config.get('cytoplasm_channels', ['GFP', 'RFP'])
        
        # Processing parameters
        self.min_cell_area = config.get('min_cell_area', 100)
        self.max_cell_area = config.get('max_cell_area', 10000)
        self.texture_distance = config.get('texture_distance', [1, 2, 3])
        self.texture_angles = config.get('texture_angles', [0, 45, 90, 135])
        
        # Multi-scale parameters
        self.scale_factors = config.get('scale_factors', [1.0, 0.5, 0.25])
        self.patch_sizes = config.get('patch_sizes', [32, 64, 128])
        
        # Initialize feature extractors
        self.morphology_extractor = MorphologyFeatureExtractor(config)
        self.intensity_extractor = IntensityFeatureExtractor(config)
        self.texture_extractor = TextureFeatureExtractor(config)
        self.spatial_extractor = SpatialFeatureExtractor(config)
        self.organelle_detector = OrganelleDetector(config)
        
        # Feature storage
        self.feature_names = []
        self.feature_statistics = {}
        
    def extract_all_features(self, 
                           image: np.ndarray,
                           segmentation_mask: Optional[np.ndarray] = None,
                           metadata: Optional[Dict] = None) -> Dict:
        """
        Extract comprehensive feature set from cellular image.
        
        Args:
            image: Multi-channel image [C, H, W]
            segmentation_mask: Cell segmentation mask [H, W]
            metadata: Additional metadata
            
        Returns:
            Dictionary containing all extracted features
        """
        
        features = {}
        
        # Generate segmentation if not provided
        if segmentation_mask is None:
            segmentation_mask = self._generate_segmentation(image)
        
        # Extract different feature types
        if self.extract_morphology:
            morphology_features = self.morphology_extractor.extract_features(
                image, segmentation_mask
            )
            features.update(morphology_features)
        
        if self.extract_intensity:
            intensity_features = self.intensity_extractor.extract_features(
                image, segmentation_mask
            )
            features.update(intensity_features)
        
        if self.extract_texture:
            texture_features = self.texture_extractor.extract_features(
                image, segmentation_mask
            )
            features.update(texture_features)
        
        if self.extract_spatial:
            spatial_features = self.spatial_extractor.extract_features(
                image, segmentation_mask
            )
            features.update(spatial_features)
        
        if self.extract_colocalization:
            colocalization_features = self._extract_colocalization_features(image)
            features.update(colocalization_features)
        
        if self.extract_organelles:
            organelle_features = self.organelle_detector.detect_organelles(
                image, segmentation_mask
            )
            features.update(organelle_features)
        
        # Multi-scale features
        multiscale_features = self._extract_multiscale_features(image, segmentation_mask)
        features.update(multiscale_features)
        
        # Add metadata features
        if metadata:
            features.update(self._extract_metadata_features(metadata))
        
        # Post-process features
        features = self._postprocess_features(features)
        
        return features
    
    def _generate_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Generate cell segmentation from nuclear channel."""
        
        # Get nuclear channel
        nuclear_idx = 0
        if self.nuclear_channel in self.channels:
            nuclear_idx = self.channels.index(self.nuclear_channel)
        
        nuclear_image = image[nuclear_idx]
        
        # Normalize
        nuclear_image = (nuclear_image - nuclear_image.min()) / (nuclear_image.max() - nuclear_image.min() + 1e-8)
        
        # Gaussian filter
        smoothed = filters.gaussian(nuclear_image, sigma=1.0)
        
        # Threshold using Otsu
        threshold = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold
        
        # Clean up binary image
        binary = morphology.remove_small_objects(binary, min_size=self.min_cell_area)
        binary = morphology.remove_small_holes(binary, area_threshold=self.min_cell_area//4)
        
        # Watershed segmentation
        distance = ndimage.distance_transform_edt(binary)
        local_maxima = feature.peak_local_maxima(distance, min_distance=10, threshold_abs=0.3*distance.max())
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(zip(*local_maxima))] = np.arange(1, len(local_maxima) + 1)
        
        segmentation = segmentation.watershed(-distance, markers, mask=binary)
        
        return segmentation
    
    def _extract_colocalization_features(self, image: np.ndarray) -> Dict:
        """Extract colocalization features between channels."""
        
        features = {}
        
        # Compute colocalization between all channel pairs
        for i in range(len(self.channels)):
            for j in range(i + 1, len(self.channels)):
                if i < image.shape[0] and j < image.shape[0]:
                    ch1_name = self.channels[i]
                    ch2_name = self.channels[j]
                    
                    ch1_data = image[i].flatten()
                    ch2_data = image[j].flatten()
                    
                    # Pearson correlation
                    correlation = np.corrcoef(ch1_data, ch2_data)[0, 1]
                    if not np.isnan(correlation):
                        features[f'colocalization_pearson_{ch1_name}_{ch2_name}'] = correlation
                    
                    # Manders coefficients
                    mask1 = ch1_data > np.percentile(ch1_data, 95)
                    mask2 = ch2_data > np.percentile(ch2_data, 95)
                    
                    if np.sum(mask1) > 0:
                        m1 = np.sum(ch2_data[mask1]) / np.sum(ch2_data)
                        features[f'colocalization_manders_m1_{ch1_name}_{ch2_name}'] = m1
                    
                    if np.sum(mask2) > 0:
                        m2 = np.sum(ch1_data[mask2]) / np.sum(ch1_data)
                        features[f'colocalization_manders_m2_{ch1_name}_{ch2_name}'] = m2
                    
                    # Overlap coefficient
                    overlap = np.sum(np.minimum(ch1_data, ch2_data)) / np.sqrt(np.sum(ch1_data**2) * np.sum(ch2_data**2) + 1e-8)
                    features[f'colocalization_overlap_{ch1_name}_{ch2_name}'] = overlap
        
        return features
    
    def _extract_multiscale_features(self, 
                                   image: np.ndarray,
                                   segmentation_mask: np.ndarray) -> Dict:
        """Extract features at multiple scales."""
        
        features = {}
        
        for scale_factor in self.scale_factors:
            if scale_factor == 1.0:
                continue
                
            # Resize image and mask
            new_size = (int(image.shape[1] * scale_factor), int(image.shape[2] * scale_factor))
            
            scaled_image = np.zeros((image.shape[0], *new_size))
            for c in range(image.shape[0]):
                scaled_image[c] = cv2.resize(image[c], new_size, interpolation=cv2.INTER_LINEAR)
            
            scaled_mask = cv2.resize(segmentation_mask.astype(np.float32), new_size, interpolation=cv2.INTER_NEAREST)
            
            # Extract basic features at this scale
            scale_features = self._extract_basic_features_for_scale(scaled_image, scaled_mask)
            
            # Add scale prefix
            for key, value in scale_features.items():
                features[f'scale_{scale_factor}_{key}'] = value
        
        return features
    
    def _extract_basic_features_for_scale(self, 
                                        image: np.ndarray,
                                        segmentation_mask: np.ndarray) -> Dict:
        """Extract basic features for a specific scale."""
        
        features = {}
        
        # Cell count and density
        labels = measure.label(segmentation_mask > 0)
        features['cell_count'] = len(np.unique(labels)) - 1  # Exclude background
        features['cell_density'] = features['cell_count'] / (image.shape[1] * image.shape[2])
        
        # Basic intensity statistics
        for c, channel_name in enumerate(self.channels):
            if c < image.shape[0]:
                channel_data = image[c]
                features[f'{channel_name}_mean'] = np.mean(channel_data)
                features[f'{channel_name}_std'] = np.std(channel_data)
                features[f'{channel_name}_median'] = np.median(channel_data)
        
        return features
    
    def _extract_metadata_features(self, metadata: Dict) -> Dict:
        """Extract features from metadata."""
        
        features = {}
        
        # Well position features
        if 'well' in metadata:
            well = metadata['well']
            if len(well) >= 2:
                row_letter = well[0]
                col_number = int(well[1:])
                
                features['well_row'] = ord(row_letter.upper()) - ord('A')
                features['well_col'] = col_number - 1
                features['well_edge_distance'] = min(features['well_row'], features['well_col'])
        
        # Plate position effects
        if 'plate_id' in metadata:
            features['plate_id_hash'] = hash(metadata['plate_id']) % 1000
        
        # Imaging parameters
        imaging_params = ['exposure_time', 'gain', 'laser_power']
        for param in imaging_params:
            if param in metadata:
                features[f'imaging_{param}'] = float(metadata[param])
        
        # Treatment information
        if 'compound' in metadata:
            features['has_compound'] = 1.0
            if 'concentration' in metadata:
                features['log_concentration'] = np.log10(float(metadata['concentration']) + 1e-9)
        else:
            features['has_compound'] = 0.0
        
        if 'gene_target' in metadata:
            features['has_gene_target'] = 1.0
        else:
            features['has_gene_target'] = 0.0
        
        return features
    
    def _postprocess_features(self, features: Dict) -> Dict:
        """Post-process extracted features."""
        
        processed_features = {}
        
        for key, value in features.items():
            # Handle NaN values
            if np.isnan(value) or np.isinf(value):
                processed_features[key] = 0.0
            else:
                processed_features[key] = float(value)
        
        return processed_features
    
    def extract_features_batch(self, 
                             images: List[np.ndarray],
                             segmentation_masks: Optional[List[np.ndarray]] = None,
                             metadata_list: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Extract features from a batch of images."""
        
        print(f"🔬 Extracting features from {len(images)} images...")
        
        all_features = []
        
        for i, image in enumerate(images):
            segmentation_mask = segmentation_masks[i] if segmentation_masks else None
            metadata = metadata_list[i] if metadata_list else {}
            
            try:
                features = self.extract_all_features(image, segmentation_mask, metadata)
                features['image_index'] = i
                all_features.append(features)
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(images)} images")
                    
            except Exception as e:
                print(f"⚠️ Error processing image {i}: {e}")
                continue
        
        if not all_features:
            print("❌ No features extracted successfully")
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Store feature names and statistics
        self.feature_names = [col for col in features_df.columns if col != 'image_index']
        self._compute_feature_statistics(features_df)
        
        print(f"✅ Feature extraction complete: {len(features_df)} samples, {len(self.feature_names)} features")
        
        return features_df
    
    def _compute_feature_statistics(self, features_df: pd.DataFrame) -> None:
        """Compute and store feature statistics."""
        
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        self.feature_statistics = {
            'means': numeric_features.mean().to_dict(),
            'stds': numeric_features.std().to_dict(),
            'medians': numeric_features.median().to_dict(),
            'mins': numeric_features.min().to_dict(),
            'maxs': numeric_features.max().to_dict(),
            'nulls': numeric_features.isnull().sum().to_dict()
        }
    
    def get_feature_importance(self, 
                             features_df: pd.DataFrame,
                             target_column: str,
                             method: str = 'mutual_info') -> pd.Series:
        """Compute feature importance scores."""
        
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        X = features_df.drop(columns=[target_column, 'image_index'], errors='ignore')
        y = features_df[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        if method == 'mutual_info':
            importance_scores = mutual_info_regression(X, y, random_state=42)
        elif method == 'f_score':
            importance_scores, _ = f_regression(X, y)
        elif method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_scores = rf.feature_importances_
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        importance_series = pd.Series(importance_scores, index=X.columns)
        return importance_series.sort_values(ascending=False)

class MorphologyFeatureExtractor:
    """Extract morphological features from cellular images."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.min_cell_area = config.get('min_cell_area', 100)
        self.max_cell_area = config.get('max_cell_area', 10000)
        
    def extract_features(self, 
                        image: np.ndarray,
                        segmentation_mask: np.ndarray) -> Dict:
        """Extract morphological features."""
        
        features = {}
        
        # Get cell regions
        labels = measure.label(segmentation_mask > 0)
        props = measure.regionprops(labels)
        
        if not props:
            return self._get_empty_features()
        
        # Filter cells by area
        valid_props = [prop for prop in props 
                      if self.min_cell_area <= prop.area <= self.max_cell_area]
        
        if not valid_props:
            return self._get_empty_features()
        
        # Extract features for each cell and aggregate
        cell_features = self._extract_single_cell_features(valid_props)
        aggregated_features = self._aggregate_cell_features(cell_features)
        
        # Add population-level features
        population_features = self._extract_population_features(valid_props, image.shape[1:])
        aggregated_features.update(population_features)
        
        return aggregated_features
    
    def _extract_single_cell_features(self, props: List) -> Dict:
        """Extract features for individual cells."""
        
        cell_data = {
            'areas': [],
            'perimeters': [],
            'eccentricities': [],
            'solidities': [],
            'aspect_ratios': [],
            'extent': [],
            'convex_areas': [],
            'euler_numbers': [],
            'orientations': [],
            'major_axis_lengths': [],
            'minor_axis_lengths': []
        }
        
        for prop in props:
            cell_data['areas'].append(prop.area)
            cell_data['perimeters'].append(prop.perimeter)
            cell_data['eccentricities'].append(prop.eccentricity)
            cell_data['solidities'].append(prop.solidity)
            
            # Aspect ratio
            if prop.minor_axis_length > 0:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            else:
                aspect_ratio = 1.0
            cell_data['aspect_ratios'].append(aspect_ratio)
            
            cell_data['extent'].append(prop.extent)
            cell_data['convex_areas'].append(prop.convex_area)
            cell_data['euler_numbers'].append(prop.euler_number)
            cell_data['orientations'].append(prop.orientation)
            cell_data['major_axis_lengths'].append(prop.major_axis_length)
            cell_data['minor_axis_lengths'].append(prop.minor_axis_length)
        
        return cell_data
    
    def _aggregate_cell_features(self, cell_data: Dict) -> Dict:
        """Aggregate single-cell features to population statistics."""
        
        features = {}
        
        for feature_name, values in cell_data.items():
            if not values:
                continue
                
            values = np.array(values)
            
            # Basic statistics
            features[f'morph_{feature_name}_mean'] = np.mean(values)
            features[f'morph_{feature_name}_std'] = np.std(values)
            features[f'morph_{feature_name}_median'] = np.median(values)
            features[f'morph_{feature_name}_mad'] = np.median(np.abs(values - np.median(values)))
            
            # Percentiles
            features[f'morph_{feature_name}_p25'] = np.percentile(values, 25)
            features[f'morph_{feature_name}_p75'] = np.percentile(values, 75)
            features[f'morph_{feature_name}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
            
            # Distribution shape
            features[f'morph_{feature_name}_skewness'] = self._compute_skewness(values)
            features[f'morph_{feature_name}_kurtosis'] = self._compute_kurtosis(values)
            
            # Coefficient of variation
            features[f'morph_{feature_name}_cv'] = np.std(values) / (np.mean(values) + 1e-8)
        
        return features
    
    def _extract_population_features(self, props: List, image_shape: Tuple) -> Dict:
        """Extract population-level morphological features."""
        
        features = {}
        
        # Cell count and density
        features['morph_cell_count'] = len(props)
        total_area = image_shape[0] * image_shape[1]
        features['morph_cell_density'] = len(props) / total_area
        
        # Total cell area fraction
        total_cell_area = sum(prop.area for prop in props)
        features['morph_cell_area_fraction'] = total_cell_area / total_area
        
        # Spatial distribution of cells
        centroids = np.array([prop.centroid for prop in props])
        
        if len(centroids) > 1:
            # Center of mass
            features['morph_center_of_mass_y'] = np.mean(centroids[:, 0]) / image_shape[0]
            features['morph_center_of_mass_x'] = np.mean(centroids[:, 1]) / image_shape[1]
            
            # Spatial spread
            features['morph_spatial_spread_y'] = np.std(centroids[:, 0]) / image_shape[0]
            features['morph_spatial_spread_x'] = np.std(centroids[:, 1]) / image_shape[1]
            
            # Spatial clustering
            from scipy.spatial.distance import pdist
            distances = pdist(centroids)
            features['morph_mean_cell_distance'] = np.mean(distances)
            features['morph_min_cell_distance'] = np.min(distances)
            
            # Spatial regularity (coefficient of variation of distances)
            features['morph_spatial_regularity'] = 1.0 / (1.0 + np.std(distances) / np.mean(distances))
        
        return features
    
    def _get_empty_features(self) -> Dict:
        """Return empty features when no valid cells found."""
        
        empty_features = {}
        
        # Define feature names that would be extracted
        feature_types = ['areas', 'perimeters', 'eccentricities', 'solidities', 
                        'aspect_ratios', 'extent', 'convex_areas', 'euler_numbers',
                        'orientations', 'major_axis_lengths', 'minor_axis_lengths']
        
        stats = ['mean', 'std', 'median', 'mad', 'p25', 'p75', 'iqr', 'skewness', 'kurtosis', 'cv']
        
        for feature_type in feature_types:
            for stat in stats:
                empty_features[f'morph_{feature_type}_{stat}'] = 0.0
        
        # Population features
        population_features = ['cell_count', 'cell_density', 'cell_area_fraction',
                             'center_of_mass_y', 'center_of_mass_x', 'spatial_spread_y',
                             'spatial_spread_x', 'mean_cell_distance', 'min_cell_distance',
                             'spatial_regularity']
        
        for feature in population_features:
            empty_features[f'morph_{feature}'] = 0.0
        
        return empty_features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
        return kurtosis

class IntensityFeatureExtractor:
    """Extract intensity-based features from cellular images."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5', 'Cy7'])
        
    def extract_features(self, 
                        image: np.ndarray,
                        segmentation_mask: np.ndarray) -> Dict:
        """Extract intensity features for all channels."""
        
        features = {}
        
        for c, channel_name in enumerate(self.channels):
            if c < image.shape[0]:
                channel_features = self._extract_channel_intensity_features(
                    image[c], segmentation_mask, channel_name
                )
                features.update(channel_features)
        
        return features
    
    def _extract_channel_intensity_features(self, 
                                          channel: np.ndarray,
                                          segmentation_mask: np.ndarray,
                                          channel_name: str) -> Dict:
        """Extract intensity features for a single channel."""
        
        features = {}
        prefix = f'intensity_{channel_name}'
        
        # Global intensity statistics
        features[f'{prefix}_global_mean'] = np.mean(channel)
        features[f'{prefix}_global_std'] = np.std(channel)
        features[f'{prefix}_global_median'] = np.median(channel)
        features[f'{prefix}_global_mad'] = np.median(np.abs(channel - np.median(channel)))
        features[f'{prefix}_global_sum'] = np.sum(channel)
        
        # Percentile-based features
        percentiles = np.percentile(channel, [1, 5, 25, 75, 95, 99])
        features[f'{prefix}_p1'] = percentiles[0]
        features[f'{prefix}_p5'] = percentiles[1]
        features[f'{prefix}_p25'] = percentiles[2]
        features[f'{prefix}_p75'] = percentiles[3]
        features[f'{prefix}_p95'] = percentiles[4]
        features[f'{prefix}_p99'] = percentiles[5]
        
        # Dynamic range
        features[f'{prefix}_dynamic_range'] = percentiles[5] - percentiles[0]  # p99 - p1
        features[f'{prefix}_iqr'] = percentiles[3] - percentiles[2]  # p75 - p25
        
        # Distribution shape
        features[f'{prefix}_skewness'] = self._compute_skewness(channel.flatten())
        features[f'{prefix}_kurtosis'] = self._compute_kurtosis(channel.flatten())
        
        # Cell-based intensity features
        if segmentation_mask is not None:
            cell_features = self._extract_cell_intensity_features(channel, segmentation_mask, prefix)
            features.update(cell_features)
        
        # Spatial intensity patterns
        spatial_features = self._extract_spatial_intensity_features(channel, prefix)
        features.update(spatial_features)
        
        return features
    
    def _extract_cell_intensity_features(self, 
                                       channel: np.ndarray,
                                       segmentation_mask: np.ndarray,
                                       prefix: str) -> Dict:
        """Extract cell-based intensity features."""
        
        features = {}
        
        # Get cell regions
        labels = measure.label(segmentation_mask > 0)
        props = measure.regionprops(labels, intensity_image=channel)
        
        if not props:
            return features
        
        # Extract intensity statistics for each cell
        cell_intensities = {
            'mean_intensities': [],
            'max_intensities': [],
            'min_intensities': [],
            'integrated_intensities': [],
            'intensity_stds': []
        }
        
        for prop in props:
            cell_intensities['mean_intensities'].append(prop.mean_intensity)
            cell_intensities['max_intensities'].append(prop.max_intensity)
            cell_intensities['min_intensities'].append(prop.min_intensity)
            cell_intensities['integrated_intensities'].append(prop.mean_intensity * prop.area)
            
            # Standard deviation within cell
            cell_pixels = channel[prop.coords[:, 0], prop.coords[:, 1]]
            cell_intensities['intensity_stds'].append(np.std(cell_pixels))
        
        # Aggregate cell intensity features
        for feature_name, values in cell_intensities.items():
            if values:
                values = np.array(values)
                features[f'{prefix}_cell_{feature_name}_mean'] = np.mean(values)
                features[f'{prefix}_cell_{feature_name}_std'] = np.std(values)
                features[f'{prefix}_cell_{feature_name}_median'] = np.median(values)
                features[f'{prefix}_cell_{feature_name}_cv'] = np.std(values) / (np.mean(values) + 1e-8)
        
        # Cell-to-cell intensity variation
        if cell_intensities['mean_intensities']:
            mean_intensities = np.array(cell_intensities['mean_intensities'])
            features[f'{prefix}_cell_intensity_heterogeneity'] = np.std(mean_intensities) / (np.mean(mean_intensities) + 1e-8)
        
        return features
    
    def _extract_spatial_intensity_features(self, channel: np.ndarray, prefix: str) -> Dict:
        """Extract spatial intensity pattern features."""
        
        features = {}
        
        # Center vs edge intensity comparison
        h, w = channel.shape
        center_h, center_w = h // 2, w // 2
        
        # Define center and edge regions
        center_size = min(h, w) // 4
        center_region = channel[
            center_h - center_size:center_h + center_size,
            center_w - center_size:center_w + center_size
        ]
        
        # Edge regions (border pixels)
        edge_width = min(h, w) // 10
        edge_region = np.concatenate([
            channel[:edge_width, :].flatten(),
            channel[-edge_width:, :].flatten(),
            channel[:, :edge_width].flatten(),
            channel[:, -edge_width:].flatten()
        ])
        
        if center_region.size > 0 and edge_region.size > 0:
            features[f'{prefix}_center_mean'] = np.mean(center_region)
            features[f'{prefix}_edge_mean'] = np.mean(edge_region)
            features[f'{prefix}_center_edge_ratio'] = np.mean(center_region) / (np.mean(edge_region) + 1e-8)
        
        # Gradient-based features
        gradient_y, gradient_x = np.gradient(channel.astype(np.float32))
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        features[f'{prefix}_gradient_mean'] = np.mean(gradient_magnitude)
        features[f'{prefix}_gradient_std'] = np.std(gradient_magnitude)
        features[f'{prefix}_gradient_max'] = np.max(gradient_magnitude)
        
        # Local variance
        from scipy.ndimage import generic_filter
        local_variance = generic_filter(channel.astype(np.float32), np.var, size=5)
        features[f'{prefix}_local_variance_mean'] = np.mean(local_variance)
        features[f'{prefix}_local_variance_std'] = np.std(local_variance)
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
        return kurtosis

class TextureFeatureExtractor:
    """Extract texture features using various methods."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5', 'Cy7'])
        self.distances = config.get('texture_distances', [1, 2, 3])
        self.angles = config.get('texture_angles', [0, np.pi/4, np.pi/2, 3*np.pi/4])
        
    def extract_features(self, 
                        image: np.ndarray,
                        segmentation_mask: np.ndarray) -> Dict:
        """Extract texture features for all channels."""
        
        features = {}
        
        for c, channel_name in enumerate(self.channels):
            if c < image.shape[0]:
                channel_features = self._extract_channel_texture_features(
                    image[c], segmentation_mask, channel_name
                )
                features.update(channel_features)
        
        return features
    
    def _extract_channel_texture_features(self, 
                                        channel: np.ndarray,
                                        segmentation_mask: np.ndarray,
                                        channel_name: str) -> Dict:
        """Extract texture features for a single channel."""
        
        features = {}
        prefix = f'texture_{channel_name}'
        
        # Convert to 8-bit for texture analysis
        channel_8bit = self._convert_to_8bit(channel)
        
        # GLCM-based texture features
        glcm_features = self._extract_glcm_features(channel_8bit, prefix)
        features.update(glcm_features)
        
        # Local Binary Pattern features
        lbp_features = self._extract_lbp_features(channel_8bit, prefix)
        features.update(lbp_features)
        
        # Gabor filter responses
        gabor_features = self._extract_gabor_features(channel, prefix)
        features.update(gabor_features)
        
        # Laws texture features
        laws_features = self._extract_laws_features(channel, prefix)
        features.update(laws_features)
        
        return features
    
    def _convert_to_8bit(self, channel: np.ndarray) -> np.ndarray:
        """Convert channel to 8-bit for texture analysis."""
        
        # Normalize to 0-255 range
        channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        return (channel_norm * 255).astype(np.uint8)
    
    def _extract_glcm_features(self, channel_8bit: np.ndarray, prefix: str) -> Dict:
        """Extract Gray Level Co-occurrence Matrix features."""
        
        from skimage.feature import graycomatrix, graycoprops
        
        features = {}
        
        try:
            # Compute GLCM
            glcm = graycomatrix(
                channel_8bit, 
                distances=self.distances, 
                angles=self.angles,
                levels=32,  # Reduce levels for computational efficiency
                symmetric=True,
                normed=True
            )
            
            # Extract texture properties
            texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            
            for prop in texture_props:
                prop_values = graycoprops(glcm, prop)
                
                # Aggregate across distances and angles
                features[f'{prefix}_glcm_{prop}_mean'] = np.mean(prop_values)
                features[f'{prefix}_glcm_{prop}_std'] = np.std(prop_values)
                features[f'{prefix}_glcm_{prop}_range'] = np.max(prop_values) - np.min(prop_values)
        
        except Exception as e:
            print(f"⚠️ Error computing GLCM features: {e}")
            # Fill with zeros if computation fails
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                features[f'{prefix}_glcm_{prop}_mean'] = 0.0
                features[f'{prefix}_glcm_{prop}_std'] = 0.0
                features[f'{prefix}_glcm_{prop}_range'] = 0.0
        
        return features
    
    def _extract_lbp_features(self, channel_8bit: np.ndarray, prefix: str) -> Dict:
        """Extract Local Binary Pattern features."""
        
        from skimage.feature import local_binary_pattern
        
        features = {}
        
        try:
            # LBP parameters
            radius = 3
            n_points = 8 * radius
            
            # Compute LBP
            lbp = local_binary_pattern(channel_8bit, n_points, radius, method='uniform')
            
            # Histogram of LBP patterns
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, density=True)
            
            # Statistical features of LBP histogram
            features[f'{prefix}_lbp_uniformity'] = np.sum(lbp_hist**2)
            features[f'{prefix}_lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            features[f'{prefix}_lbp_mean'] = np.mean(lbp)
            features[f'{prefix}_lbp_std'] = np.std(lbp)
            
            # Dominant patterns
            dominant_patterns = np.argsort(lbp_hist)[-3:]  # Top 3 patterns
            for i, pattern_idx in enumerate(dominant_patterns):
                features[f'{prefix}_lbp_dominant_pattern_{i}'] = lbp_hist[pattern_idx]
        
        except Exception as e:
            print(f"⚠️ Error computing LBP features: {e}")
            # Fill with zeros if computation fails
            lbp_feature_names = ['uniformity', 'entropy', 'mean', 'std'] + [f'dominant_pattern_{i}' for i in range(3)]
            for feature_name in lbp_feature_names:
                features[f'{prefix}_lbp_{feature_name}'] = 0.0
        
        return features
    
    def _extract_gabor_features(self, channel: np.ndarray, prefix: str) -> Dict:
        """Extract Gabor filter response features."""
        
        from skimage.filters import gabor
        
        features = {}
        
        # Gabor filter parameters
        frequencies = [0.1, 0.3, 0.5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            gabor_responses = []
            
            for frequency in frequencies:
                for angle in angles:
                    # Apply Gabor filter
                    filtered_real, filtered_imag = gabor(channel, frequency=frequency, theta=angle)
                    
                    # Compute magnitude
                    magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
                    gabor_responses.append(magnitude)
            
            # Extract statistics from Gabor responses
            for i, response in enumerate(gabor_responses):
                freq_idx = i // len(angles)
                angle_idx = i % len(angles)
                
                response_prefix = f'{prefix}_gabor_f{freq_idx}_a{angle_idx}'
                features[f'{response_prefix}_mean'] = np.mean(response)
                features[f'{response_prefix}_std'] = np.std(response)
                features[f'{response_prefix}_energy'] = np.sum(response**2)
            
            # Overall Gabor features
            all_responses = np.array(gabor_responses)
            features[f'{prefix}_gabor_overall_mean'] = np.mean(all_responses)
            features[f'{prefix}_gabor_overall_std'] = np.std(all_responses)
            features[f'{prefix}_gabor_max_response'] = np.max(all_responses)
        
        except Exception as e:
            print(f"⚠️ Error computing Gabor features: {e}")
            # Fill with zeros if computation fails
            for i in range(len(frequencies) * len(angles)):
                freq_idx = i // len(angles)
                angle_idx = i % len(angles)
                response_prefix = f'{prefix}_gabor_f{freq_idx}_a{angle_idx}'
                for stat in ['mean', 'std', 'energy']:
                    features[f'{response_prefix}_{stat}'] = 0.0
            
            for stat in ['overall_mean', 'overall_std', 'max_response']:
                features[f'{prefix}_gabor_{stat}'] = 0.0
        
        return features
    
    def _extract_laws_features(self, channel: np.ndarray, prefix: str) -> Dict:
        """Extract Laws texture features."""
        
        features = {}
        
        try:
            # Laws filter masks
            L5 = np.array([1, 4, 6, 4, 1])  # Level
            E5 = np.array([-1, -2, 0, 2, 1])  # Edge
            S5 = np.array([-1, 0, 2, 0, -1])  # Spot
            R5 = np.array([1, -4, 6, -4, 1])  # Ripple
            
            # Create 2D filters by outer products
            filters = {
                'L5L5': np.outer(L5, L5),
                'L5E5': np.outer(L5, E5),
                'E5L5': np.outer(E5, L5),
                'E5E5': np.outer(E5, E5),
                'S5S5': np.outer(S5, S5),
                'R5R5': np.outer(R5, R5),
                'L5S5': np.outer(L5, S5),
                'S5L5': np.outer(S5, L5),
                'E5S5': np.outer(E5, S5),
                'S5E5': np.outer(S5, E5)
            }
            
            # Apply filters and compute texture energy
            for filter_name, filter_mask in filters.items():
                filtered = cv2.filter2D(channel.astype(np.float32), -1, filter_mask)
                texture_energy = np.mean(np.abs(filtered))
                features[f'{prefix}_laws_{filter_name.lower()}_energy'] = texture_energy
        
        except Exception as e:
            print(f"⚠️ Error computing Laws features: {e}")
            # Fill with zeros if computation fails
            filter_names = ['l5l5', 'l5e5', 'e5l5', 'e5e5', 's5s5', 'r5r5', 'l5s5', 's5l5', 'e5s5', 's5e5']
            for filter_name in filter_names:
                features[f'{prefix}_laws_{filter_name}_energy'] = 0.0
        
        return features

class SpatialFeatureExtractor:
    """Extract spatial organization features."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP', 'Cy5', 'Cy7'])
        
    def extract_features(self, 
                        image: np.ndarray,
                        segmentation_mask: np.ndarray) -> Dict:
        """Extract spatial features."""
        
        features = {}
        
        # Get cell regions
        labels = measure.label(segmentation_mask > 0)
        props = measure.regionprops(labels)
        
        if not props:
            return self._get_empty_spatial_features()
        
        # Extract spatial organization features
        spatial_features = self._extract_spatial_organization_features(props, image.shape[1:])
        features.update(spatial_features)
        
        # Extract nearest neighbor features
        nn_features = self._extract_nearest_neighbor_features(props)
        features.update(nn_features)
        
        # Extract clustering features
        clustering_features = self._extract_clustering_features(props, image.shape[1:])
        features.update(clustering_features)
        
        return features
    
    def _extract_spatial_organization_features(self, props: List, image_shape: Tuple) -> Dict:
        """Extract spatial organization features."""
        
        features = {}
        
        if not props:
            return features
        
        # Get centroids
        centroids = np.array([prop.centroid for prop in props])
        
        # Spatial distribution metrics
        features['spatial_cell_count'] = len(centroids)
        features['spatial_density'] = len(centroids) / (image_shape[0] * image_shape[1])
        
        if len(centroids) > 1:
            # Center of mass
            com_y, com_x = np.mean(centroids, axis=0)
            features['spatial_center_of_mass_y'] = com_y / image_shape[0]
            features['spatial_center_of_mass_x'] = com_x / image_shape[1]
            
            # Spatial spread
            features['spatial_spread_y'] = np.std(centroids[:, 0]) / image_shape[0]
            features['spatial_spread_x'] = np.std(centroids[:, 1]) / image_shape[1]
            
            # Elliptical fit to cell positions
            try:
                from sklearn.covariance import EllipticEnvelope
                ee = EllipticEnvelope(contamination=0.1)
                ee.fit(centroids)
                
                # Extract ellipse parameters
                covariance = ee.covariance_
                eigenvals, eigenvecs = np.linalg.eigh(covariance)
                
                features['spatial_ellipse_major_axis'] = np.sqrt(eigenvals[1])
                features['spatial_ellipse_minor_axis'] = np.sqrt(eigenvals[0])
                features['spatial_ellipse_eccentricity'] = np.sqrt(1 - eigenvals[0] / eigenvals[1]) if eigenvals[1] > 0 else 0
                features['spatial_ellipse_orientation'] = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
            
            except Exception:
                features['spatial_ellipse_major_axis'] = 0.0
                features['spatial_ellipse_minor_axis'] = 0.0
                features['spatial_ellipse_eccentricity'] = 0.0
                features['spatial_ellipse_orientation'] = 0.0
        
        return features
    
    def _extract_nearest_neighbor_features(self, props: List) -> Dict:
        """Extract nearest neighbor features."""
        
        features = {}
        
        if len(props) < 2:
            return self._get_empty_nn_features()
        
        centroids = np.array([prop.centroid for prop in props])
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(centroids))
        
        # Remove diagonal (zero distances)
        np.fill_diagonal(distances, np.inf)
        
        # Nearest neighbor distances
        nn_distances = np.min(distances, axis=1)
        
        features['spatial_nn_mean_distance'] = np.mean(nn_distances)
        features['spatial_nn_std_distance'] = np.std(nn_distances)
        features['spatial_nn_median_distance'] = np.median(nn_distances)
        features['spatial_nn_min_distance'] = np.min(nn_distances)
        features['spatial_nn_max_distance'] = np.max(nn_distances)
        
        # Nearest neighbor regularity
        features['spatial_nn_regularity'] = 1.0 / (1.0 + np.std(nn_distances) / np.mean(nn_distances))
        
        # k-nearest neighbor features (k=3, 5)
        for k in [3, 5]:
            if len(props) > k:
                k_nn_distances = np.sort(distances, axis=1)[:, :k]
                mean_k_nn = np.mean(k_nn_distances)
                features[f'spatial_{k}nn_mean_distance'] = mean_k_nn
                features[f'spatial_{k}nn_std_distance'] = np.std(k_nn_distances)
        
        return features
    
    def _extract_clustering_features(self, props: List, image_shape: Tuple) -> Dict:
        """Extract clustering features."""
        
        features = {}
        
        if len(props) < 3:
            return self._get_empty_clustering_features()
        
        centroids = np.array([prop.centroid for prop in props])
        
        # Hopkins statistic for clustering tendency
        hopkins_stat = self._compute_hopkins_statistic(centroids, image_shape)
        features['spatial_hopkins_statistic'] = hopkins_stat
        
        # Ripley's K function (simplified)
        ripley_k = self._compute_ripley_k(centroids, image_shape)
        features['spatial_ripley_k'] = ripley_k
        
        # Morisita index for spatial dispersion
        morisita_index = self._compute_morisita_index(centroids, image_shape)
        features['spatial_morisita_index'] = morisita_index
        
        # Quadrat analysis
        quadrat_features = self._compute_quadrat_analysis(centroids, image_shape)
        features.update(quadrat_features)
        
        return features
    
    def _compute_hopkins_statistic(self, centroids: np.ndarray, image_shape: Tuple) -> float:
        """Compute Hopkins statistic for clustering tendency."""
        
        try:
            n_sample = min(len(centroids) // 2, 50)  # Sample size
            
            # Random points within image bounds
            random_points = np.random.rand(n_sample, 2) * np.array([image_shape[0], image_shape[1]])
            
            # Distances from random points to nearest data points
            from scipy.spatial.distance import cdist
            rand_distances = np.min(cdist(random_points, centroids), axis=1)
            
            # Distances from sample data points to nearest other data points
            sample_indices = np.random.choice(len(centroids), n_sample, replace=False)
            sample_points = centroids[sample_indices]
            
            other_points = np.delete(centroids, sample_indices, axis=0)
            if len(other_points) > 0:
                data_distances = np.min(cdist(sample_points, other_points), axis=1)
            else:
                data_distances = np.ones(n_sample)
            
            # Hopkins statistic
            u_sum = np.sum(rand_distances)
            w_sum = np.sum(data_distances)
            hopkins = u_sum / (u_sum + w_sum)
            
            return hopkins
        
        except Exception:
            return 0.5  # Random distribution
    
    def _compute_ripley_k(self, centroids: np.ndarray, image_shape: Tuple) -> float:
        """Compute simplified Ripley's K function."""
        
        try:
            # Choose radius as fraction of image diagonal
            radius = np.sqrt(image_shape[0]**2 + image_shape[1]**2) * 0.1
            
            # Count neighbors within radius for each point
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(centroids))
            
            neighbor_counts = np.sum(distances <= radius, axis=1) - 1  # Exclude self
            mean_neighbors = np.mean(neighbor_counts)
            
            # Expected number of neighbors for random distribution
            area = image_shape[0] * image_shape[1]
            density = len(centroids) / area
            expected_neighbors = density * np.pi * radius**2
            
            # Ripley's K (normalized)
            ripley_k = mean_neighbors / expected_neighbors if expected_neighbors > 0 else 1.0
            
            return ripley_k
        
        except Exception:
            return 1.0  # Random distribution
    
    def _compute_morisita_index(self, centroids: np.ndarray, image_shape: Tuple) -> float:
        """Compute Morisita index for spatial dispersion."""
        
        try:
            # Divide image into quadrats
            n_quadrats = 16  # 4x4 grid
            quadrat_size_y = image_shape[0] / 4
            quadrat_size_x = image_shape[1] / 4
            
            # Count points in each quadrat
            quadrat_counts = np.zeros(n_quadrats)
            
            for i, (y, x) in enumerate(centroids):
                quadrat_y = min(int(y // quadrat_size_y), 3)
                quadrat_x = min(int(x // quadrat_size_x), 3)
                quadrat_idx = quadrat_y * 4 + quadrat_x
                quadrat_counts[quadrat_idx] += 1
            
            # Morisita index calculation
            n = len(centroids)
            if n <= 1:
                return 1
            sum_ni_ni_minus_1 = np.sum(quadrat_counts * (quadrat_counts - 1))
            
            morisita_index = n_quadrats * sum_ni_ni_minus_1 / (n * (n - 1))
            
            return morisita_index
        
        except Exception:
            return 1.0  # Random distribution
    
    def _compute_quadrat_analysis(self, centroids: np.ndarray, image_shape: Tuple) -> Dict:
        """Compute quadrat analysis features."""
        
        features = {}
        
        try:
            # Different quadrat sizes
            quadrat_sizes = [2, 4, 8]  # Grid sizes: 2x2, 4x4, 8x8
            
            for grid_size in quadrat_sizes:
                quadrat_size_y = image_shape[0] / grid_size
                quadrat_size_x = image_shape[1] / grid_size
                
                # Count points in each quadrat
                quadrat_counts = np.zeros(grid_size * grid_size)
                
                for y, x in centroids:
                    quadrat_y = min(int(y // quadrat_size_y), grid_size - 1)
                    quadrat_x = min(int(x // quadrat_size_x), grid_size - 1)
                    quadrat_idx = quadrat_y * grid_size + quadrat_x
                    quadrat_counts[quadrat_idx] += 1
                
                # Variance-to-mean ratio
                mean_count = np.mean(quadrat_counts)
                var_count = np.var(quadrat_counts)
                
                if mean_count > 0:
                    vm_ratio = var_count / mean_count
                else:
                    vm_ratio = 1.0
                
                features[f'spatial_quadrat_{grid_size}x{grid_size}_vm_ratio'] = vm_ratio
                features[f'spatial_quadrat_{grid_size}x{grid_size}_mean_count'] = mean_count
                features[f'spatial_quadrat_{grid_size}x{grid_size}_max_count'] = np.max(quadrat_counts)
                features[f'spatial_quadrat_{grid_size}x{grid_size}_empty_fraction'] = np.sum(quadrat_counts == 0) / len(quadrat_counts)
        
        except Exception:
            for grid_size in [2, 4, 8]:
                features[f'spatial_quadrat_{grid_size}x{grid_size}_vm_ratio'] = 1.0
                features[f'spatial_quadrat_{grid_size}x{grid_size}_mean_count'] = 0.0
                features[f'spatial_quadrat_{grid_size}x{grid_size}_max_count'] = 0.0
                features[f'spatial_quadrat_{grid_size}x{grid_size}_empty_fraction'] = 1.0
        
        return features
    
    def _get_empty_spatial_features(self) -> Dict:
        """Return empty spatial features when no cells found."""
        
        features = {}
        
        # Basic spatial features
        spatial_features = ['cell_count', 'density', 'center_of_mass_y', 'center_of_mass_x',
                          'spread_y', 'spread_x', 'ellipse_major_axis', 'ellipse_minor_axis',
                          'ellipse_eccentricity', 'ellipse_orientation']
        
        for feature in spatial_features:
            features[f'spatial_{feature}'] = 0.0
        
        # Nearest neighbor features
        nn_features = self._get_empty_nn_features()
        features.update(nn_features)
        
        # Clustering features
        clustering_features = self._get_empty_clustering_features()
        features.update(clustering_features)
        
        return features
    
    def _get_empty_nn_features(self) -> Dict:
        """Return empty nearest neighbor features."""
        
        features = {}
        
        nn_feature_names = ['mean_distance', 'std_distance', 'median_distance', 
                          'min_distance', 'max_distance', 'regularity']
        
        for feature in nn_feature_names:
            features[f'spatial_nn_{feature}'] = 0.0
        
        # k-NN features
        for k in [3, 5]:
            features[f'spatial_nn_k{k}_mean_distance'] = 0.0
            features[f'spatial_nn_k{k}_std_distance'] = 0.0
        
        return features
    
    def _get_empty_clustering_features(self) -> Dict:
        """Return empty clustering features."""
        
        features = {}
        
        clustering_feature_names = ['hopkins_statistic', 'ripley_k', 'morisita_index']
        
        for feature in clustering_feature_names:
            features[f'spatial_{feature}'] = 0.0
        
        # Quadrat analysis features
        for grid_size in [2, 4, 8]:
            features[f'spatial_quadrat_{grid_size}x{grid_size}_vm_ratio'] = 0.0
            features[f'spatial_quadrat_{grid_size}x{grid_size}_mean_count'] = 0.0
            features[f'spatial_quadrat_{grid_size}x{grid_size}_max_count'] = 0.0
            features[f'spatial_quadrat_{grid_size}x{grid_size}_empty_fraction'] = 1.0
        
        return features