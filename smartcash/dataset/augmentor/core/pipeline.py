"""
File: smartcash/dataset/augmentor/core/pipeline.py
Deskripsi: Pipeline factory untuk augmentasi - refactored dengan one-liner utilities
"""

import albumentations as A
from typing import Dict, Any, List, Callable, Optional
from smartcash.common.logger import get_logger

class PipelineFactory:
    """Factory untuk membuat pipeline augmentasi dengan konfigurasi fleksibel"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        
    def create_pipeline(self, aug_type: str = "combined", intensity: float = 0.5) -> A.Compose:
        """Factory method untuk membuat pipeline augmentasi"""
        return self._get_pipeline_builder(aug_type)(intensity)
    
    def _get_pipeline_builder(self, aug_type: str) -> Callable:
        """Get pipeline builder function berdasarkan tipe"""
        builders = {
            'position': self._build_position_pipeline,      # 🎯 Penelitian: variasi posisi
            'lighting': self._build_lighting_pipeline,      # 🎯 Penelitian: variasi pencahayaan
            'combined': self._build_combined_pipeline,      # 🎯 Default: posisi + pencahayaan
            # Individual transformations untuk fleksibilitas masa depan
            'geometric': self._build_geometric_pipeline,
            'color': self._build_color_pipeline, 
            'noise': self._build_noise_pipeline,
            'light': self._build_light_pipeline,
            'heavy': self._build_heavy_pipeline
        }
        return builders.get(aug_type, self._build_combined_pipeline)
    
    def _build_position_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline untuk variasi posisi uang kertas"""
        return A.Compose([
            A.Rotate(limit=int(15 * intensity), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1 * intensity, 
                scale_limit=0.05 * intensity, 
                rotate_limit=12 * intensity, 
                p=0.7
            ),
            A.Perspective(scale=(0.02 * intensity, 0.08 * intensity), p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_lighting_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline untuk variasi pencahayaan uang kertas"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.25 * intensity, 
                contrast_limit=0.2 * intensity, 
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.6),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),
            A.CLAHE(clip_limit=2.0 * intensity, tile_grid_size=(8, 8), p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_combined_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline kombinasi posisi + pencahayaan (default untuk penelitian)"""
        return A.Compose([
            # Variasi posisi
            A.Rotate(limit=int(12 * intensity), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08 * intensity, 
                scale_limit=0.04 * intensity, 
                rotate_limit=10 * intensity, 
                p=0.6
            ),
            A.Perspective(scale=(0.02 * intensity, 0.06 * intensity), p=0.3),
            
            # Variasi pencahayaan
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * intensity, 
                contrast_limit=0.15 * intensity, 
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.5),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=1,
                shadow_dimension=4,
                p=0.2
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Individual transformations untuk konfigurasi fleksibel
    def _build_geometric_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline geometri individual"""
        return A.Compose([
            A.Rotate(limit=int(20 * intensity), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1 * intensity, scale_limit=0.1 * intensity, rotate_limit=15 * intensity, p=0.7),
            A.Perspective(scale=(0.05 * intensity, 0.1 * intensity), p=0.3)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_color_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline color individual"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3 * intensity, contrast_limit=0.3 * intensity, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(20 * intensity), sat_shift_limit=int(30 * intensity), val_shift_limit=int(20 * intensity), p=0.6),
            A.CLAHE(clip_limit=3.0 * intensity, tile_grid_size=(8, 8), p=0.5),
            A.ColorJitter(brightness=0.2 * intensity, contrast=0.2 * intensity, saturation=0.2 * intensity, hue=0.1 * intensity, p=0.5)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_noise_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline noise individual"""
        return A.Compose([
            A.GaussNoise(var_limit=(10 * intensity, 50 * intensity), p=0.5),
            A.ISONoise(color_shift=(0.01 * intensity, 0.05 * intensity), intensity=(0.1 * intensity, 0.5 * intensity), p=0.3),
            A.Blur(blur_limit=int(3 * intensity), p=0.3),
            A.MotionBlur(blur_limit=int(7 * intensity), p=0.3)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_light_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline ringan"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1 * intensity, contrast_limit=0.1 * intensity, p=0.6),
            A.Rotate(limit=int(5 * intensity), p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_heavy_pipeline(self, intensity: float) -> A.Compose:
        """Pipeline berat"""
        return A.Compose([
            A.Rotate(limit=int(25 * intensity), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.15 * intensity, scale_limit=0.15 * intensity, rotate_limit=20 * intensity, p=0.8),
            A.Perspective(scale=(0.05 * intensity, 0.15 * intensity), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4 * intensity, contrast_limit=0.4 * intensity, p=0.9),
            A.GaussNoise(var_limit=(15 * intensity, 75 * intensity), p=0.6),
            A.Blur(blur_limit=int(5 * intensity), p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# One-liner utilities untuk pipeline creation (focused pada penelitian)
create_augmentation_pipeline = lambda config, aug_type='combined', intensity=0.5: PipelineFactory(config).create_pipeline(aug_type, intensity)
get_research_pipeline_types = lambda: ['position', 'lighting', 'combined']  # 🎯 Pipeline untuk penelitian
get_available_pipeline_types = lambda: ['position', 'lighting', 'combined', 'geometric', 'color', 'noise', 'light', 'heavy']
validate_intensity = lambda intensity: max(0.1, min(1.0, intensity))
get_pipeline_description = lambda aug_type: {
    'position': '📍 Variasi posisi uang kertas (rotasi, flip, perspektif)',
    'lighting': '💡 Variasi pencahayaan uang kertas (brightness, shadow, gamma)',
    'combined': '🎯 Kombinasi posisi + pencahayaan (default penelitian)',
    'geometric': '🔄 Transformasi geometri lengkap',
    'color': '🎨 Transformasi warna lengkap',
    'noise': '📡 Noise dan blur effects',
    'light': '🌟 Augmentasi ringan',  
    'heavy': '⚡ Augmentasi berat'
}.get(aug_type, '🎯 Pipeline augmentasi default')