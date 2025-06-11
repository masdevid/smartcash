"""
File: smartcash/dataset/preprocessor/utils/normalization.py
Deskripsi: Normalisasi YOLO-specific dengan resize dan padding untuk object detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

from smartcash.common.logger import get_logger

class YOLONormalizer:
    """🎯 YOLO-specific normalization engine untuk object detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # YOLO settings
        self.target_size = tuple(self.config.get('target_size', [640, 640]))
        self.preserve_aspect_ratio = self.config.get('preserve_aspect_ratio', True)
        self.pad_color = self.config.get('pad_color', 114)  # Gray padding
        self.normalize_pixels = self.config.get('normalize_pixel_values', True)
    
    def preprocess_for_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """🎯 Complete YOLO preprocessing dengan metadata"""
        try:
            original_shape = image.shape[:2]  # (h, w)
            
            # Resize dengan aspect ratio preservation
            if self.preserve_aspect_ratio:
                resized, scale_info = self._resize_with_padding(image)
            else:
                resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                scale_info = {'scale': min(self.target_size[0] / original_shape[1], 
                                         self.target_size[1] / original_shape[0]),
                            'pad_x': 0, 'pad_y': 0}
            
            # Normalize pixels untuk neural network
            if self.normalize_pixels:
                normalized = resized.astype(np.float32) / 255.0
            else:
                normalized = resized.astype(np.float32)
            
            # Metadata untuk coordinate transformation
            metadata = {
                'original_shape': original_shape,
                'target_shape': self.target_size,
                'scale_info': scale_info,
                'normalized': self.normalize_pixels
            }
            
            return normalized, metadata
            
        except Exception as e:
            self.logger.error(f"❌ YOLO preprocessing error: {str(e)}")
            # Fallback: simple resize dan normalize
            resized = cv2.resize(image, self.target_size)
            normalized = resized.astype(np.float32) / 255.0 if self.normalize_pixels else resized.astype(np.float32)
            return normalized, {'error': str(e)}
    
    def _resize_with_padding(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """🖼️ Resize dengan padding untuk maintain aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale untuk fit dalam target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), self.pad_color, dtype=image.dtype)
        else:
            padded = np.full((target_h, target_w), self.pad_color, dtype=image.dtype)
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        scale_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'new_size': (new_w, new_h)
        }
        
        return padded, scale_info
    
    def transform_bbox_coordinates(self, bboxes: np.ndarray, scale_info: Dict[str, Any], 
                                 reverse: bool = False) -> np.ndarray:
        """🔧 Transform bbox coordinates untuk/dari normalized space"""
        if len(bboxes) == 0:
            return bboxes
        
        try:
            scale = scale_info['scale']
            pad_x = scale_info['pad_x']
            pad_y = scale_info['pad_y']
            
            transformed = bboxes.copy()
            
            if not reverse:
                # Transform dari original ke normalized space
                # YOLO format: [class, x_center, y_center, width, height] (normalized)
                if transformed.shape[1] >= 5:
                    # Apply scale dan padding
                    transformed[:, 1] = (transformed[:, 1] * scale + pad_x) / self.target_size[0]
                    transformed[:, 2] = (transformed[:, 2] * scale + pad_y) / self.target_size[1]
                    transformed[:, 3] = transformed[:, 3] * scale / self.target_size[0]
                    transformed[:, 4] = transformed[:, 4] * scale / self.target_size[1]
            else:
                # Transform dari normalized ke original space
                if transformed.shape[1] >= 5:
                    # Reverse padding dan scale
                    transformed[:, 1] = (transformed[:, 1] * self.target_size[0] - pad_x) / scale
                    transformed[:, 2] = (transformed[:, 2] * self.target_size[1] - pad_y) / scale
                    transformed[:, 3] = transformed[:, 3] * self.target_size[0] / scale
                    transformed[:, 4] = transformed[:, 4] * self.target_size[1] / scale
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"❌ Bbox transform error: {str(e)}")
            return bboxes

# === FACTORY FUNCTIONS ===

def create_yolo_normalizer(config: Dict[str, Any] = None) -> YOLONormalizer:
    """🏭 Factory untuk create YOLO normalizer"""
    return YOLONormalizer(config)

# === CONVENIENCE FUNCTIONS ===

def preprocess_image_for_yolo(image: np.ndarray, target_size: Tuple[int, int] = (640, 640),
                             preserve_aspect: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """🎯 One-liner YOLO preprocessing"""
    config = {'target_size': target_size, 'preserve_aspect_ratio': preserve_aspect}
    normalizer = create_yolo_normalizer(config)
    return normalizer.preprocess_for_yolo(image)

def normalize_yolo_safe(image: np.ndarray) -> np.ndarray:
    """🎯 One-liner safe YOLO normalization"""
    try:
        normalized, _ = preprocess_image_for_yolo(image)
        return normalized
    except Exception:
        return image.astype(np.float32) / 255.0