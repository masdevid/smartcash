"""
File: smartcash/utils/augmentation/augmentation_pipeline.py
Author: Perbaikan untuk warnings
Deskripsi: Pipeline transformasi untuk augmentasi dataset dengan perbaikan warnings
"""

import albumentations as A
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

class AugmentationPipeline:
    """Pipeline untuk transformasi augmentasi."""
    
    def __init__(self, config: Dict, output_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi pipeline augmentasi.
        
        Args:
            config: Konfigurasi augmentasi
            output_dir: Direktori output
            logger: Logger
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.transforms = {}
        
        # Build transformasi
        self.build_transforms()
    
    def build_transforms(self):
        """Build semua transformasi dari config."""
        # Ambil konfigurasi augmentasi
        aug_config = self.config.get('augmentation', {})
        
        # Transformasi posisi
        self.transforms['position'] = self.build_position_transforms(aug_config)
        
        # Transformasi pencahayaan
        self.transforms['lighting'] = self.build_lighting_transforms(aug_config)
        
        # Transformasi kombinasi
        self.transforms['combined'] = self.build_combined_transforms(aug_config)
        
        # Transformasi ekstrim
        self.transforms['extreme_rotation'] = self.build_extreme_transforms(aug_config)
    
    def build_position_transforms(self, aug_config: Dict) -> A.Compose:
        """
        Build transformasi posisi.
        
        Args:
            aug_config: Konfigurasi augmentasi
            
        Returns:
            Transformasi untuk posisi
        """
        position_cfg = aug_config.get('position', {})
        
        # Gunakan Affine daripada ShiftScaleRotate
        return A.Compose([
            A.HorizontalFlip(p=position_cfg.get('fliplr', 0.5)),
            A.Affine(
                scale={"x": (1 - position_cfg.get('scale', 0.1), 1 + position_cfg.get('scale', 0.1)),
                       "y": (1 - position_cfg.get('scale', 0.1), 1 + position_cfg.get('scale', 0.1))},
                translate_percent={"x": (-position_cfg.get('translate', 0.1), position_cfg.get('translate', 0.1)),
                                 "y": (-position_cfg.get('translate', 0.1), position_cfg.get('translate', 0.1))},
                rotate=(-position_cfg.get('degrees', 10), position_cfg.get('degrees', 10)),
                p=position_cfg.get('rotation_prob', 0.5)
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def build_lighting_transforms(self, aug_config: Dict) -> A.Compose:
        """
        Build transformasi pencahayaan.
        
        Args:
            aug_config: Konfigurasi augmentasi
            
        Returns:
            Transformasi untuk pencahayaan
        """
        lighting_cfg = aug_config.get('lighting', {})
        
        return A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=lighting_cfg.get('hsv_h', 0.015) * 360,
                sat_shift_limit=lighting_cfg.get('hsv_s', 0.7) * 100,
                val_shift_limit=lighting_cfg.get('hsv_v', 0.4) * 100,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=lighting_cfg.get('brightness', 0.3),
                contrast_limit=lighting_cfg.get('contrast', 0.3),
                p=lighting_cfg.get('brightness_prob', 0.5)
            ),
            # Fix untuk ImageCompression warning
            A.ImageCompression(
                quality_lower=80,
                quality_upper=100,
                p=lighting_cfg.get('compress', 0.2)
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def build_combined_transforms(self, aug_config: Dict) -> A.Compose:
        """
        Build transformasi kombinasi.
        
        Args:
            aug_config: Konfigurasi augmentasi
            
        Returns:
            Transformasi kombinasi
        """
        position_cfg = aug_config.get('position', {})
        lighting_cfg = aug_config.get('lighting', {})
        
        return A.Compose([
            # Kombinasi posisi
            A.HorizontalFlip(p=position_cfg.get('fliplr', 0.5)),
            A.Affine(
                scale={"x": (1 - position_cfg.get('scale', 0.1), 1 + position_cfg.get('scale', 0.1)),
                       "y": (1 - position_cfg.get('scale', 0.1), 1 + position_cfg.get('scale', 0.1))},
                translate_percent={"x": (-position_cfg.get('translate', 0.1), position_cfg.get('translate', 0.1)),
                                 "y": (-position_cfg.get('translate', 0.1), position_cfg.get('translate', 0.1))},
                rotate=(-position_cfg.get('degrees', 10), position_cfg.get('degrees', 10)),
                p=0.5
            ),
            
            # Kombinasi pencahayaan
            A.HueSaturationValue(
                hue_shift_limit=lighting_cfg.get('hsv_h', 0.015) * 360,
                sat_shift_limit=lighting_cfg.get('hsv_s', 0.7) * 100,
                val_shift_limit=lighting_cfg.get('hsv_v', 0.4) * 100,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=lighting_cfg.get('brightness', 0.3),
                contrast_limit=lighting_cfg.get('contrast', 0.3),
                p=0.3
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def build_extreme_transforms(self, aug_config: Dict) -> A.Compose:
        """
        Build transformasi ekstrim untuk dataset lebih robust.
        
        Args:
            aug_config: Konfigurasi augmentasi
            
        Returns:
            Transformasi ekstrim
        """
        extreme_cfg = aug_config.get('extreme', {})
        min_angle = extreme_cfg.get('rotation_min', 30)
        max_angle = extreme_cfg.get('rotation_max', 90)
        prob = extreme_cfg.get('probability', 0.3)
        
        return A.Compose([
            A.Affine(
                rotate=(-max_angle, max_angle),
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                p=prob
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.4, 
                contrast_limit=0.4,
                p=prob
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def get_transform(self, aug_type: str) -> A.Compose:
        """
        Dapatkan transformasi berdasarkan tipe.
        
        Args:
            aug_type: Tipe augmentasi
            
        Returns:
            Transformasi Albumentations
        """
        return self.transforms.get(aug_type, self.transforms.get('combined', None))
    
    def transform_image(self, 
                       image: np.ndarray, 
                       bboxes: Optional[List] = None, 
                       class_labels: Optional[List] = None, 
                       aug_type: str = 'combined') -> Dict:
        """
        Transformasi gambar dan bboxes dengan tipe augmentasi tertentu.
        
        Args:
            image: Array gambar
            bboxes: List bounding boxes [x, y, w, h]
            class_labels: List label kelas
            aug_type: Tipe augmentasi
            
        Returns:
            Dict hasil transformasi
        """
        transform = self.get_transform(aug_type)
        if transform is None:
            if self.logger:
                self.logger.warning(f"⚠️ Transformasi '{aug_type}' tidak ditemukan, menggunakan default")
            transform = self.transforms.get('combined')
        
        # Setup data untuk transformasi
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = [0] * len(bboxes)
            
        # Eksekusi transformasi
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat transformasi: {str(e)}")
            return {'image': image, 'bboxes': bboxes, 'class_labels': class_labels}