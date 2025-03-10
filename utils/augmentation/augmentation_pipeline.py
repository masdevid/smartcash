"""
File: smartcash/utils/augmentation/augmentation_pipeline.py
Author: Alfrida Sabar
Deskripsi: Definisi pipeline augmentasi dengan berbagai metode transformasi gambar
"""

import albumentations as A
from typing import Dict, Optional

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase

class AugmentationPipeline(AugmentationBase):
    """Definisi pipeline augmentasi dengan berbagai metode transformasi gambar."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi pipeline augmentasi."""
        super().__init__(config, output_dir, logger)
        self._setup_augmentation_pipelines()
    
    def _setup_augmentation_pipelines(self) -> None:
        """Setup pipeline augmentasi berdasarkan konfigurasi."""
        aug_config = self.config.get('training', {})
        
        bbox_params = A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
        
        self.position_aug = A.Compose([
            A.SafeRotate(limit=30, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=aug_config.get('translate', 0.1),
                scale_limit=aug_config.get('scale', 0.5),
                rotate_limit=aug_config.get('degrees', 45),
                p=0.5
            )
        ], bbox_params=bbox_params)
        
        self.lighting_aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=aug_config.get('hsv_h', 0.015),
                sat_shift_limit=aug_config.get('hsv_s', 0.7),
                val_shift_limit=aug_config.get('hsv_v', 0.4),
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=80,
                quality_upper=100,
                p=0.2
            )
        ], bbox_params=bbox_params)
        
        self.combined_aug = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=30, p=0.7),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=aug_config.get('translate', 0.1),
                    scale_limit=aug_config.get('scale', 0.5),
                    rotate_limit=aug_config.get('degrees', 45),
                    p=0.5
                )
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomShadow(p=0.5),
                A.HueSaturationValue(p=0.5)
            ], p=0.7),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.3)),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.2)
            ], p=0.2)
        ], bbox_params=bbox_params)
        
        self.extreme_rotation_aug = A.Compose([
            A.RandomRotate90(p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=90,
                p=0.8
            ),
            A.RandomBrightnessContrast(p=0.3)
        ], bbox_params=bbox_params)
        
        self.logger.info("✅ Pipeline augmentasi siap: 4 variasi tersedia")
    
    def get_pipeline(self, augmentation_type: str):
        """Dapatkan pipeline augmentasi berdasarkan jenis."""
        pipelines = {
            'position': self.position_aug,
            'lighting': self.lighting_aug,
            'extreme_rotation': self.extreme_rotation_aug
        }
        return pipelines.get(augmentation_type, self.combined_aug)