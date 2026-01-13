"""
Feature extraction module for vehicle classification.
Extracts hand-crafted features from vehicle images for traditional ML classifiers.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
from skimage.feature import hog, local_binary_pattern
from skimage import color
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class VehicleFeatureExtractor:
    """Extract multiple types of features from vehicle images."""
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (224, 224),
                 hog_orientations: int = 9,
                 hog_pixels_per_cell: Tuple[int, int] = (8, 8),
                 hog_cells_per_block: Tuple[int, int] = (2, 2),
                 color_bins: int = 32,
                 lbp_points: int = 24,
                 lbp_radius: int = 3):
        """
        Initialize feature extractor with parameters.
        
        Args:
            img_size: Resize images to this size (width, height)
            hog_orientations: Number of HOG orientation bins
            hog_pixels_per_cell: Size of HOG cells
            hog_cells_per_block: Number of cells per HOG block
            color_bins: Number of bins for color histograms
            lbp_points: Number of points for LBP
            lbp_radius: Radius for LBP
        """
        self.img_size = img_size
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.color_bins = color_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        
        # Calculate feature dimensions for verification
        self.feature_dims = self._calculate_feature_dimensions()
        
    def _calculate_feature_dimensions(self) -> Dict[str, int]:
        """Calculate expected feature dimensions."""
        # HOG features
        hog_blocks_x = (self.img_size[0] // self.hog_pixels_per_cell[0]) - self.hog_cells_per_block[0] + 1
        hog_blocks_y = (self.img_size[1] // self.hog_pixels_per_cell[1]) - self.hog_cells_per_block[1] + 1
        hog_dim = hog_blocks_x * hog_blocks_y * self.hog_cells_per_block[0] * self.hog_cells_per_block[1] * self.hog_orientations
        
        # Color histograms (RGB + HSV)
        color_dim = self.color_bins * 3 * 2  # 3 channels for RGB, 3 for HSV
        
        # LBP histogram
        lbp_dim = self.lbp_points + 2  # uniform patterns + 2
        
        # Statistical features (12 stats × 3 channels × 2 color spaces)
        stats_dim = 12 * 3 * 2
        
        # Edge features
        edge_dim = 10
        
        return {
            'hog': hog_dim,
            'color_hist': color_dim,
            'lbp': lbp_dim,
            'statistical': stats_dim,
            'edge': edge_dim,
            'total': hog_dim + color_dim + lbp_dim + stats_dim + edge_dim
        }
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features.
        HOG captures edge directions, useful for vehicle shape/silhouette.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            1D HOG feature vector
        """
        # Convert to grayscale
        gray = color.rgb2gray(image)
        
        # Extract HOG features
        features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        return features
    
    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histograms from RGB and HSV color spaces.
        Color helps distinguish vehicle types (trucks vs cars, buses, etc.).
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Concatenated color histogram features
        """
        features = []
        
        # RGB histograms
        for channel in range(3):
            hist, _ = np.histogram(
                image[:, :, channel],
                bins=self.color_bins,
                range=(0, 1)
            )
            hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize
            features.append(hist)
        
        # HSV histograms
        hsv = color.rgb2hsv(image)
        for channel in range(3):
            hist, _ = np.histogram(
                hsv[:, :, channel],
                bins=self.color_bins,
                range=(0, 1)
            )
            hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize
            features.append(hist)
        
        return np.concatenate(features)
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) texture features.
        Texture helps distinguish vehicle surfaces and details.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            LBP histogram features
        """
        # Convert to grayscale
        gray = color.rgb2gray(image)
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_points,
            R=self.lbp_radius,
            method='uniform'
        )
        
        # Compute histogram
        n_bins = self.lbp_points + 2  # uniform patterns + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins)
        )
        hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from image channels.
        Includes mean, std, skewness, kurtosis, min, max, etc.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Statistical feature vector
        """
        features = []
        
        # RGB statistics
        for channel in range(3):
            ch_data = image[:, :, channel].ravel()
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                stats.skew(ch_data),
                stats.kurtosis(ch_data),
                np.median(ch_data),
                np.min(ch_data),
                np.max(ch_data),
                np.percentile(ch_data, 25),
                np.percentile(ch_data, 75),
                np.ptp(ch_data),  # peak-to-peak (range)
                stats.iqr(ch_data),  # interquartile range
                stats.variation(ch_data) if np.mean(ch_data) != 0 else 0  # coefficient of variation
            ])
        
        # HSV statistics
        hsv = color.rgb2hsv(image)
        for channel in range(3):
            ch_data = hsv[:, :, channel].ravel()
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                stats.skew(ch_data),
                stats.kurtosis(ch_data),
                np.median(ch_data),
                np.min(ch_data),
                np.max(ch_data),
                np.percentile(ch_data, 25),
                np.percentile(ch_data, 75),
                np.ptp(ch_data),
                stats.iqr(ch_data),
                stats.variation(ch_data) if np.mean(ch_data) != 0 else 0
            ])
        
        return np.array(features)
    
    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edge-based features using Canny edge detection.
        Edges help identify vehicle boundaries and structural elements.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Edge feature vector
        """
        # Convert to grayscale
        gray = (color.rgb2gray(image) * 255).astype(np.uint8)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge statistics
        edge_density = np.sum(edges > 0) / edges.size
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        
        # Edge distribution in quadrants (helps capture vehicle orientation)
        h, w = edges.shape
        h_mid, w_mid = h // 2, w // 2
        
        quadrants = [
            edges[:h_mid, :w_mid],      # top-left
            edges[:h_mid, w_mid:],      # top-right
            edges[h_mid:, :w_mid],      # bottom-left
            edges[h_mid:, w_mid:],      # bottom-right
        ]
        
        quadrant_densities = [np.sum(q > 0) / q.size for q in quadrants]
        
        # Horizontal and vertical edge projections
        horiz_proj = np.sum(edges, axis=1).max() / w
        vert_proj = np.sum(edges, axis=0).max() / h
        
        features = [
            edge_density,
            edge_mean,
            edge_std,
            *quadrant_densities,
            horiz_proj,
            vert_proj
        ]
        
        return np.array(features)
    
    def extract_all_features(self, image_path: Path) -> np.ndarray:
        """
        Extract all feature types from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete feature vector
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.img_size)
        
        # Normalize to [0, 1]
        image = image.astype(float) / 255.0
        
        # Extract all features
        hog_feat = self.extract_hog_features(image)
        color_feat = self.extract_color_histogram(image)
        lbp_feat = self.extract_lbp_features(image)
        stat_feat = self.extract_statistical_features(image)
        edge_feat = self.extract_edge_features(image)
        
        # Concatenate all features
        features = np.concatenate([
            hog_feat,
            color_feat,
            lbp_feat,
            stat_feat,
            edge_feat
        ])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Generate feature names for all extracted features.
        
        Returns:
            List of feature names
        """
        names = []
        
        # HOG feature names
        for i in range(self.feature_dims['hog']):
            names.append(f'hog_{i:04d}')
        
        # Color histogram names
        color_channels = ['R', 'G', 'B', 'H', 'S', 'V']
        for ch_idx, ch_name in enumerate(color_channels):
            for bin_idx in range(self.color_bins):
                names.append(f'color_{ch_name}_bin_{bin_idx:02d}')
        
        # LBP names
        for i in range(self.feature_dims['lbp']):
            names.append(f'lbp_bin_{i:02d}')
        
        # Statistical feature names
        stat_names = ['mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max', 
                     'q25', 'q75', 'range', 'iqr', 'coef_var']
        color_spaces = ['RGB', 'HSV']
        channels = ['ch0', 'ch1', 'ch2']
        for space in color_spaces:
            for ch in channels:
                for stat in stat_names:
                    names.append(f'stat_{space}_{ch}_{stat}')
        
        # Edge feature names
        edge_names = ['edge_density', 'edge_mean', 'edge_std',
                     'edge_quad_tl', 'edge_quad_tr', 'edge_quad_bl', 'edge_quad_br',
                     'edge_horiz_proj', 'edge_vert_proj']
        names.extend(edge_names)
        
        return names
    
    def print_feature_summary(self):
        """Print summary of feature dimensions."""
        print("="*60)
        print("FEATURE EXTRACTOR CONFIGURATION")
        print("="*60)
        print(f"Image size: {self.img_size}")
        print(f"\nFeature Dimensions:")
        print(f"  HOG features:         {self.feature_dims['hog']:5d}")
        print(f"  Color histograms:     {self.feature_dims['color_hist']:5d}")
        print(f"  LBP features:         {self.feature_dims['lbp']:5d}")
        print(f"  Statistical features: {self.feature_dims['statistical']:5d}")
        print(f"  Edge features:        {self.feature_dims['edge']:5d}")
        print(f"  {'─'*40}")
        print(f"  TOTAL:                {self.feature_dims['total']:5d}")
        print("="*60)


def extract_features_from_directory(data_dir: Path,
                                    extractor: VehicleFeatureExtractor,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from all images in a directory structure.
    
    Args:
        data_dir: Root directory (train/val/test)
        extractor: VehicleFeatureExtractor instance
        verbose: Print progress
        
    Returns:
        features: (N, D) array of features
        labels: (N,) array of class indices
        image_paths: List of image paths
    """
    features_list = []
    labels_list = []
    image_paths = []
    
    # Get all class folders
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
    
    if verbose:
        print(f"\nExtracting features from: {data_dir.name}")
        print(f"Found {len(class_folders)} classes")
    
    # Process each class
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        label = class_to_idx[class_name]
        
        # Get all images in class
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [img for img in class_folder.iterdir() 
                 if img.suffix.lower() in img_extensions]
        
        if verbose:
            print(f"  Processing {class_name}: {len(images)} images...", end=' ')
        
        # Extract features from each image
        for img_path in images:
            try:
                features = extractor.extract_all_features(img_path)
                features_list.append(features)
                labels_list.append(label)
                image_paths.append(str(img_path))
            except Exception as e:
                print(f"\n    Warning: Failed to process {img_path.name}: {e}")
                continue
        
        if verbose:
            print("✓")
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    if verbose:
        print(f"\nExtracted features: {features_array.shape}")
        print(f"Labels: {labels_array.shape}")
        print(f"Feature dimensionality: {features_array.shape[1]}")
    
    return features_array, labels_array, image_paths


if __name__ == "__main__":
    # Test feature extraction
    print("Testing VehicleFeatureExtractor...")
    
    extractor = VehicleFeatureExtractor(img_size=(224, 224))
    extractor.print_feature_summary()
    
    print("\nFeature names sample (first 10):")
    feature_names = extractor.get_feature_names()
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i:4d}: {name}")
    print(f"  ... ({len(feature_names) - 10} more features)")
