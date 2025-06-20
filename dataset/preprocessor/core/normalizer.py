"""
File: smartcash/dataset/preprocessor/core/normalizer.py
Deskripsi: YOLOv5-compatible normalization engine untuk object detection
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path

from smartcash.common.logger import get_logger

class YOLONormalizer:
    """🎯 YOLOv5-compatible normalization dengan reusable API"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # YOLO normalization parameters
        self.target_size = tuple(self.config.get('target_size', [640, 640]))
        self.pixel_range = self.config.get('pixel_range', [0, 1])
        self.preserve_aspect_ratio = self.config.get('preserve_aspect_ratio', True)
        self.pad_color = self.config.get('pad_color', 114)
        self.interpolation = self._get_interpolation_method(self.config.get('interpolation', 'linear'))
        
        # Performance options
        self.batch_processing = self.config.get('batch_processing', False)
    
    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """🎯 Main normalization method untuk single image"""
        try:
            original_shape = image.shape[:2]  # (h, w)
            
            # Resize dengan aspect ratio preservation
            if self.preserve_aspect_ratio:
                resized_image, transform_info = self._resize_with_padding(image)
            else:
                resized_image = cv2.resize(image, self.target_size, interpolation=self.interpolation)
                transform_info = self._calculate_simple_transform(original_shape)
            
            # Normalize pixel values untuk neural network
            normalized = self._normalize_pixels(resized_image)
            
            # Metadata untuk inference dan visualization
            metadata = {
                'original_shape': original_shape,
                'target_shape': self.target_size,
                'transform_info': transform_info,
                'normalization': {
                    'pixel_range': self.pixel_range,
                    'method': 'yolo_v5'
                }
            }
            
            return normalized, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Normalization error: {str(e)}")
            raise ValueError(f"Normalization failed: {str(e)}")
    
    def denormalize(self, normalized_image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """🔄 Denormalize untuk visualization"""
        try:
            # Reverse pixel normalization
            denorm_pixels = self._denormalize_pixels(normalized_image)
            
            # Remove padding dan restore original aspect ratio
            transform_info = metadata.get('transform_info', {})
            original_shape = metadata.get('original_shape', (640, 640))
            
            if self.preserve_aspect_ratio and 'scale' in transform_info:
                restored = self._remove_padding_and_resize(denorm_pixels, transform_info, original_shape)
            else:
                restored = cv2.resize(denorm_pixels, (original_shape[1], original_shape[0]))
            
            return restored.astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Denormalization warning: {str(e)}")
            # Fallback: simple pixel denormalization
            return self._denormalize_pixels(normalized_image).astype(np.uint8)
    
    def batch_normalize(self, images: list) -> Tuple[np.ndarray, list]:
        """📦 Batch normalization untuk multiple images"""
        if not self.batch_processing:
            return self._sequential_normalize(images)
        
        normalized_batch = []
        metadata_batch = []
        
        for image in images:
            norm_img, metadata = self.normalize(image)
            normalized_batch.append(norm_img)
            metadata_batch.append(metadata)
        
        return np.array(normalized_batch), metadata_batch
    
    def transform_coordinates(self, coordinates: np.ndarray, metadata: Dict[str, Any], 
                            reverse: bool = False) -> np.ndarray:
        """🔧 Transform YOLO coordinates untuk normalized space"""
        if len(coordinates) == 0:
            return coordinates
        
        try:
            transform_info = metadata['transform_info']
            
            if not reverse:
                # Original → Normalized
                return self._apply_coordinate_transform(coordinates, transform_info)
            else:
                # Normalized → Original  
                return self._reverse_coordinate_transform(coordinates, transform_info)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Coordinate transform warning: {str(e)}")
            return coordinates
    
    # === PRIVATE METHODS ===
    
    def _resize_with_padding(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """🖼️ Resize dengan padding untuk maintain aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale dan new dimensions
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), self.pad_color, dtype=image.dtype)
        else:
            padded = np.full((target_h, target_w), self.pad_color, dtype=image.dtype)
        
        # Place resized image di center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        transform_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'new_size': (new_w, new_h),
            'method': 'padding'
        }
        
        return padded, transform_info
    
    def _calculate_simple_transform(self, original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """📏 Calculate transform info untuk simple resize"""
        h, w = original_shape
        target_w, target_h = self.target_size
        
        return {
            'scale_x': target_w / w,
            'scale_y': target_h / h,
            'pad_x': 0,
            'pad_y': 0,
            'method': 'resize'
        }
    
    def _normalize_pixels(self, image: np.ndarray) -> np.ndarray:
        """🎨 Normalize pixel values"""
        min_val, max_val = self.pixel_range
        normalized = image.astype(np.float32)
        
        if max_val <= 1.0:
            # Normalize to [0, 1] range
            normalized = normalized / 255.0
        
        # Apply custom range jika diperlukan
        if min_val != 0 or max_val != 1:
            normalized = normalized * (max_val - min_val) + min_val
        
        return normalized
    
    def _denormalize_pixels(self, normalized: np.ndarray) -> np.ndarray:
        """🎨 Reverse pixel normalization"""
        min_val, max_val = self.pixel_range
        
        # Reverse custom range
        if min_val != 0 or max_val != 1:
            denorm = (normalized - min_val) / (max_val - min_val)
        else:
            denorm = normalized
        
        # Scale back to [0, 255]
        if max_val <= 1.0:
            denorm = denorm * 255.0
        
        return np.clip(denorm, 0, 255)
    
    def _remove_padding_and_resize(self, padded_image: np.ndarray, 
                                  transform_info: Dict[str, Any], 
                                  original_shape: Tuple[int, int]) -> np.ndarray:
        """📐 Remove padding dan resize back ke original dimensions"""
        pad_x = transform_info['pad_x']
        pad_y = transform_info['pad_y']
        new_w, new_h = transform_info['new_size']
        
        # Extract resized image dari padded version
        extracted = padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
        
        # Resize back ke original dimensions
        orig_h, orig_w = original_shape
        restored = cv2.resize(extracted, (orig_w, orig_h), interpolation=self.interpolation)
        
        return restored
    
    def _apply_coordinate_transform(self, coords: np.ndarray, transform_info: Dict[str, Any]) -> np.ndarray:
        """🔧 Apply coordinate transformation"""
        if transform_info['method'] == 'padding':
            scale = transform_info['scale']
            pad_x = transform_info['pad_x']
            pad_y = transform_info['pad_y']
            
            transformed = coords.copy()
            if transformed.shape[1] >= 5:  # YOLO format: [class, x, y, w, h]
                transformed[:, 1] = (transformed[:, 1] * scale + pad_x) / self.target_size[0]
                transformed[:, 2] = (transformed[:, 2] * scale + pad_y) / self.target_size[1]
                transformed[:, 3] = transformed[:, 3] * scale / self.target_size[0]
                transformed[:, 4] = transformed[:, 4] * scale / self.target_size[1]
        else:
            # Simple resize transformation
            scale_x = transform_info['scale_x']
            scale_y = transform_info['scale_y']
            
            transformed = coords.copy()
            if transformed.shape[1] >= 5:
                transformed[:, 1] = transformed[:, 1] * scale_x
                transformed[:, 2] = transformed[:, 2] * scale_y
                transformed[:, 3] = transformed[:, 3] * scale_x
                transformed[:, 4] = transformed[:, 4] * scale_y
        
        return transformed
    
    def _reverse_coordinate_transform(self, coords: np.ndarray, transform_info: Dict[str, Any]) -> np.ndarray:
        """🔄 Reverse coordinate transformation"""
        if transform_info['method'] == 'padding':
            scale = transform_info['scale']
            pad_x = transform_info['pad_x']
            pad_y = transform_info['pad_y']
            
            transformed = coords.copy()
            if transformed.shape[1] >= 5:
                transformed[:, 1] = (transformed[:, 1] * self.target_size[0] - pad_x) / scale
                transformed[:, 2] = (transformed[:, 2] * self.target_size[1] - pad_y) / scale
                transformed[:, 3] = transformed[:, 3] * self.target_size[0] / scale
                transformed[:, 4] = transformed[:, 4] * self.target_size[1] / scale
        else:
            # Reverse simple resize
            scale_x = transform_info['scale_x']
            scale_y = transform_info['scale_y']
            
            transformed = coords.copy()
            if transformed.shape[1] >= 5:
                transformed[:, 1] = transformed[:, 1] / scale_x
                transformed[:, 2] = transformed[:, 2] / scale_y
                transformed[:, 3] = transformed[:, 3] / scale_x
                transformed[:, 4] = transformed[:, 4] / scale_y
        
        return transformed
    
    def _get_interpolation_method(self, method: str) -> int:
        """🔧 Map interpolation method name to OpenCV constant"""
        mapping = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        return mapping.get(method, cv2.INTER_LINEAR)
    
    def _sequential_normalize(self, images: list) -> Tuple[list, list]:
        """🔄 Sequential normalization fallback"""
        normalized_list, metadata_list = [], []
        for image in images:
            norm_img, metadata = self.normalize(image)
            normalized_list.append(norm_img)
            metadata_list.append(metadata)
        return normalized_list, metadata_list