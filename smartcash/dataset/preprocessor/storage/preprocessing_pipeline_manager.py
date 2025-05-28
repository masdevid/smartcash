"""
File: smartcash/dataset/preprocessor/storage/preprocessing_pipeline_manager.py
Deskripsi: Fixed preprocessing pipeline dengan missing methods dan reduced debug flooding
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE


class PreprocessingPipelineManager:
    """Fixed preprocessing pipeline dengan complete method set."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize pipeline dengan configuration."""
        self.config = config
        self.logger = logger or get_logger()
        
        # Extract preprocessing options
        preprocessing_config = config.get('preprocessing', {})
        self.default_options = {
            'img_size': preprocessing_config.get('img_size', DEFAULT_IMG_SIZE),
            'normalize': preprocessing_config.get('normalize', True),
            'preserve_aspect_ratio': preprocessing_config.get('preserve_aspect_ratio', True),
            'normalization_method': preprocessing_config.get('normalization_method', 'minmax')
        }
        
        self.current_options = self.default_options.copy()
    
    def set_options(self, **options) -> None:
        """FIXED: Missing method - set pipeline options."""
        self.current_options.update(options)
    
    def update_pipeline_options(self, **options) -> None:
        """Update pipeline options dengan new parameters."""
        self.set_options(**options)
    
    def process(self, image: np.ndarray, custom_options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """FIXED: Missing method - alias untuk process_image."""
        return self.process_image(image, custom_options)
    
    def process_image(self, image: np.ndarray, custom_options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Process image melalui preprocessing pipeline."""
        if image is None:
            raise ValueError("Input image is None")
        
        # Use custom options atau current options
        options = {**self.current_options, **(custom_options or {})}
        
        # Apply transformations sequentially
        processed_image = image.copy()
        
        # 1. Resize transformation
        processed_image = self._apply_resize_transformation(processed_image, options)
        
        # 2. Normalization transformation
        if options.get('normalize', True):
            processed_image = self._apply_normalization_transformation(processed_image, options)
        
        return processed_image
    
    def _apply_resize_transformation(self, image: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """Apply resize transformation dengan aspect ratio handling."""
        img_size = options.get('img_size', DEFAULT_IMG_SIZE)
        preserve_aspect = options.get('preserve_aspect_ratio', True)
        
        # Normalize img_size ke tuple
        if isinstance(img_size, int):
            target_size = (img_size, img_size)
        else:
            target_size = tuple(img_size[:2])
        
        if preserve_aspect:
            return self._resize_with_aspect_ratio(image, target_size)
        else:
            return cv2.resize(image, target_size)
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize dengan preserved aspect ratio dan padding."""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate optimal scale
        scale = min(target_width / width, target_height / height)
        new_width, new_height = int(width * scale), int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create padded canvas
        if len(image.shape) == 3:
            canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Center image dalam canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
        else:
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def _apply_normalization_transformation(self, image: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """Apply normalization berdasarkan method yang dipilih."""
        method = options.get('normalization_method', 'minmax')
        
        # Convert to float32 for processing
        normalized = image.astype(np.float32)
        
        if method == 'minmax' or method == 'none':
            # Min-max normalization [0, 1]
            if image.dtype == np.uint8:
                normalized = normalized / 255.0
        elif method == 'standard':
            # Z-score normalization (mean=0, std=1)
            if image.dtype == np.uint8:
                normalized = normalized / 255.0
            mean = np.mean(normalized)
            std = np.std(normalized)
            if std > 0:
                normalized = (normalized - mean) / std
        
        return normalized
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration information."""
        return {
            'current_options': self.current_options.copy(),
            'default_options': self.default_options.copy(),
            'supported_methods': ['minmax', 'standard', 'none'],
            'pipeline_ready': True
        }
    
    def reset_pipeline_options(self) -> None:
        """Reset pipeline options ke default."""
        self.current_options = self.default_options.copy()
    
    def validate_image_input(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate input image untuk processing."""
        validation = {'valid': True, 'issues': [], 'image_info': {}}
        
        if image is None:
            validation['valid'] = False
            validation['issues'].append('Image is None')
            return validation
        
        # Check image properties
        validation['image_info'] = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'channels': len(image.shape),
            'size': image.size
        }
        
        # Validate dimensions
        if len(image.shape) not in [2, 3]:
            validation['valid'] = False
            validation['issues'].append('Image must be 2D or 3D array')
        
        # Validate data type
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            validation['issues'].append(f'Unsupported dtype: {image.dtype}')
        
        return validation