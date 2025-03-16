"""
File: smartcash/dataset/services/preprocessor/pipeline.py
Deskripsi: Pipeline transformasi untuk preprocessing dataset yang dapat dikonfigurasi
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

from smartcash.model.utils.preprocessing_model_utils import ModelPreprocessor, letterbox
from smartcash.common.logger import get_logger


class PreprocessingPipeline:
    """
    Pipeline transformasi yang dapat dikonfigurasi untuk preprocessing dataset.
    Mendukung berbagai strategi transformasi dan preprocessing.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi preprocessing pipeline.
        
        Args:
            config: Konfigurasi pipeline
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger("preprocessing_pipeline")
        
        # Default config
        self.default_config = {
            'img_size': (640, 640),
            'use_letterbox': True,
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean
            'std': [0.229, 0.224, 0.225],   # ImageNet std
            'to_rgb': True,
            'keep_ratio': True,
            'auto_pad': True
        }
        
        # Merge konfigurasi
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Inisialisasi model preprocessor untuk transformasi yang lebih rumit
        self.model_preprocessor = ModelPreprocessor(
            img_size=self.config['img_size'],
            mean=self.config['mean'],
            std=self.config['std'],
            pad_to_square=self.config['auto_pad']
        )
        
        # Setup pipeline transformasi
        self.setup_pipeline()
        
        self.logger.info(f"🔄 Preprocessing Pipeline diinisialisasi (size: {self.config['img_size']})")
    
    def setup_pipeline(self) -> None:
        """Setup urutan transformasi dalam pipeline."""
        self.transforms = []
        
        # 1. Konversi BGR ke RGB jika diperlukan
        if self.config['to_rgb']:
            self.transforms.append(self.bgr_to_rgb)
        
        # 2. Resize dengan letterbox atau langsung
        if self.config['use_letterbox']:
            self.transforms.append(self.apply_letterbox)
        else:
            self.transforms.append(self.resize_direct)
        
        # 3. Normalisasi jika diperlukan
        if self.config['normalize']:
            self.transforms.append(self.normalize)
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Proses gambar melalui pipeline transformasi.
        
        Args:
            image: Gambar input (BGR/OpenCV format)
            
        Returns:
            Tuple (gambar yang sudah diproses, metadata transformasi)
        """
        img = image.copy()
        metadata = {
            'original_size': image.shape[:2][::-1],  # (width, height)
            'transformation_log': []
        }
        
        for transform in self.transforms:
            transform_name = transform.__name__
            img, transform_meta = transform(img)
            metadata['transformation_log'].append({
                'transform': transform_name,
                **transform_meta
            })
        
        # Tambahkan informasi terakhir
        metadata['final_size'] = img.shape[:2][::-1]  # (width, height)
        
        return img, metadata
    
    # Fungsi transformasi individual
    
    def bgr_to_rgb(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Konversi BGR ke RGB."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), {'color_conversion': 'BGR→RGB'}
        return image, {'color_conversion': 'none'}
    
    def apply_letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize gambar dengan letterbox untuk mempertahankan aspect ratio."""
        img_resized, ratio, pad = letterbox(
            image, 
            self.config['img_size'], 
            auto=self.config['auto_pad'], 
            stride=32
        )
        
        metadata = {
            'method': 'letterbox',
            'ratio': ratio,
            'padding': pad
        }
        
        return img_resized, metadata
    
    def resize_direct(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize gambar langsung ke ukuran target."""
        original_h, original_w = image.shape[:2]
        target_w, target_h = self.config['img_size']
        
        img_resized = cv2.resize(image, (target_w, target_h))
        
        metadata = {
            'method': 'direct_resize',
            'ratio': (target_w / original_w, target_h / original_h),
            'padding': (0, 0)
        }
        
        return img_resized, metadata
    
    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalisasi gambar ke rentang 0-1 dan gunakan mean/std."""
        # Konversi ke float32 dan normalisasi ke 0-1
        img_norm = image.astype(np.float32) / 255.0
        
        # Normalisasi dengan mean dan std jika konfigurasi mengaktifkannya
        if self.config.get('use_imagenet_norm', True):
            mean = np.array(self.config['mean']).reshape(1, 1, 3)
            std = np.array(self.config['std']).reshape(1, 1, 3)
            img_norm = (img_norm - mean) / std
        
        metadata = {
            'method': 'normalization',
            'range': '0-1',
            'imagenet_norm': self.config.get('use_imagenet_norm', True)
        }
        
        return img_norm, metadata
    
    @classmethod
    def create_training_pipeline(cls, img_size=(640, 640)) -> 'PreprocessingPipeline':
        """Buat pipeline khusus untuk training."""
        config = {
            'img_size': img_size,
            'use_letterbox': True,
            'normalize': True,
            'use_imagenet_norm': True
        }
        
        return cls(config=config)
    
    @classmethod
    def create_inference_pipeline(cls, img_size=(640, 640)) -> 'PreprocessingPipeline':
        """Buat pipeline khusus untuk inferensi."""
        config = {
            'img_size': img_size,
            'use_letterbox': True,
            'normalize': True,
            'use_imagenet_norm': True
        }
        
        return cls(config=config)