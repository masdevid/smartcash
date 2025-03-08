"""
File: smartcash/utils/augmentation/augmentation_pipeline.py
Author: Alfrida Sabar
Deskripsi: Definisi pipeline augmentasi dengan berbagai metode transformasi gambar
"""

import albumentations as A
import threading
from typing import Dict, Optional, Any

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase

class AugmentationPipeline(AugmentationBase):
    """
    Definisi pipeline augmentasi dengan berbagai metode transformasi gambar.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi pipeline augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori output
            logger: Logger kustom
        """
        super().__init__(config, output_dir, logger)
        
        # Setup pipeline augmentasi
        self._setup_augmentation_pipelines()
    
    def _setup_augmentation_pipelines(self) -> None:
        """Setup pipeline augmentasi berdasarkan konfigurasi."""
        # Ekstrak parameter dari config
        aug_config = self.config.get('training', {})
        
        # Pipeline posisi - variasi posisi/orientasi uang
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
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline pencahayaan - variasi cahaya dan kontras
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
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline kombinasi - gabungan posisi dan pencahayaan
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
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline khusus untuk rotasi ekstrim
        self.extreme_rotation_aug = A.Compose([
            A.RandomRotate90(p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=90,
                p=0.8
            ),
            A.RandomBrightnessContrast(p=0.3)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.2
        ))
        
        self.logger.info(f"✅ Pipeline augmentasi siap: 4 variasi tersedia")
    
    def get_pipeline(self, augmentation_type: str):
        """
        Dapatkan pipeline augmentasi berdasarkan jenis.
        
        Args:
            augmentation_type: Jenis augmentasi ('position', 'lighting', 'combined', 'extreme_rotation')
            
        Returns:
            Pipeline augmentasi yang sesuai
        """
        if augmentation_type == 'position':
            return self.position_aug
        elif augmentation_type == 'lighting':
            return self.lighting_aug
        elif augmentation_type == 'extreme_rotation':
            return self.extreme_rotation_aug
        else:  # default: combined
            return self.combined_aug