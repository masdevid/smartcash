"""
File: smartcash/dataset/services/preprocessor/pipeline.py
Deskripsi: Pipeline untuk preprocessing gambar dengan berbagai transformasi
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
import logging
from dataset.utils.dataset_constants import DEFAULT_IMG_SIZE

class PreprocessingPipeline:
    """Pipeline yang dapat dikonfigurasi untuk preprocessing gambar dengan berbagai transformasi."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Inisialisasi pipeline preprocessing.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging (opsional)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Ekstrak konfigurasi preprocessing
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Ekstrak normalisasi config
        normalization_config = preprocessing_config.get('normalization', {})
        
        # Default options
        self.options = {
            'img_size': preprocessing_config.get('img_size', DEFAULT_IMG_SIZE),
            'normalize': normalization_config.get('enabled', True),
            'preserve_aspect_ratio': normalization_config.get('preserve_aspect_ratio', True),
            'pixel_range': normalization_config.get('pixel_range', [0, 1])
        }
    
    def set_options(self, **kwargs):
        """
        Update opsi preprocessing.
        
        Args:
            **kwargs: Parameter untuk preprocessing
        """
        self.options.update(kwargs)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Jalankan preprocessing pada gambar.
        
        Args:
            image: Gambar input dalam format numpy array
            
        Returns:
            Gambar yang telah dipreprocess
        """
        if image is None:
            raise ValueError("Gambar input tidak valid (None)")
            
        # Dapatkan opsi preprocessing
        img_size = self.options.get('img_size', DEFAULT_IMG_SIZE)
        normalize = self.options.get('normalize', True)
        preserve_aspect_ratio = self.options.get('preserve_aspect_ratio', True)
        pixel_range = self.options.get('pixel_range', [0, 1])
        
        # Standarisasi image_size ke tuple atau list 2 elemen
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        # Resize gambar
        if preserve_aspect_ratio:
            processed_image = self._resize_with_aspect_ratio(image, img_size)
        else:
            processed_image = cv2.resize(image, (img_size[0], img_size[1]))
        
        # Normalisasi jika diperlukan
        if normalize:
            processed_image = self._normalize_image(processed_image, pixel_range)
        
        return processed_image
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, target_size: List[int]) -> np.ndarray:
        """
        Resize gambar dengan mempertahankan aspect ratio.
        
        Args:
            image: Gambar input
            target_size: Ukuran target [width, height]
            
        Returns:
            Gambar yang telah diresize
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Hitung scale factor
        scale_width = target_width / width
        scale_height = target_height / height
        scale = min(scale_width, scale_height)
        
        # Hitung new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize gambar
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Buat canvas kosong dengan ukuran target
        if len(image.shape) == 3:
            # RGB image
            canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            # Grayscale image
            canvas = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Hitung posisi tengah
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Tempatkan gambar resized di tengah canvas
        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_image
        else:
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
        
        return canvas
    
    def _normalize_image(self, image: np.ndarray, pixel_range: List[float] = [0, 1]) -> np.ndarray:
        """
        Normalisasi gambar ke range tertentu.
        
        Args:
            image: Gambar input
            pixel_range: Range normalisasi, default [0, 1]
            
        Returns:
            Gambar yang telah dinormalisasi
        """
        min_val, max_val = pixel_range
        
        # Convert to float32 untuk precision
        normalized = image.astype(np.float32)
        
        # Jika gambar sudah dalam range [min_val, max_val], tidak perlu diubah
        if image.dtype == np.float32 and image.max() <= max_val and image.min() >= min_val:
            return normalized
        
        # Jika gambar dalam range [0, 255], normalisasi ke [min_val, max_val]
        if image.dtype == np.uint8 or image.max() > 1.0:
            normalized = normalized / 255.0
            
        # Scale ke range yang diinginkan
        if min_val != 0 or max_val != 1:
            range_size = max_val - min_val
            normalized = normalized * range_size + min_val
            
        return normalized