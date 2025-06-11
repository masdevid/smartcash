"""
File: smartcash/dataset/augmentor/core/pipeline_factory.py
Deskripsi: Factory untuk augmentation pipelines dengan default types (lighting, position, combined)
"""

import albumentations as A
from typing import Dict, Any, Callable

class PipelineFactory:
    """🏭 Factory untuk augmentation pipelines dengan default research types"""
    
    DEFAULT_TYPES = ['lighting', 'position', 'combined']
    
    def __init__(self, config: Dict[str, Any] = None):
        from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config
        
        if config is None:
            from smartcash.dataset.augmentor.utils.config_validator import get_default_augmentation_config
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        self.aug_config = self.config.get('augmentation', {})
        
        # Load parameters dari config
        self.parameters = self.aug_config.get('parameters', {})
    
    def create_pipeline(self, aug_type: str = None, intensity: float = None) -> A.Compose:
        """Create pipeline dengan config defaults"""
        # Use config defaults jika tidak disediakan
        aug_type = aug_type or self.aug_config.get('types', ['combined'])[0]
        intensity = intensity if intensity is not None else self.aug_config.get('intensity', 0.7)
        
        # Validate intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Get pipeline builder
        builder = self._get_pipeline_builder(aug_type)
        return builder(intensity)
    
    def get_available_types(self) -> list:
        """Get available pipeline types"""
        return self.DEFAULT_TYPES + ['geometric', 'color', 'noise']
    
    def _get_pipeline_builder(self, aug_type: str) -> Callable:
        """Get pipeline builder function"""
        builders = {
            'lighting': self._build_lighting_pipeline,
            'position': self._build_position_pipeline,
            'combined': self._build_combined_pipeline,
            'geometric': self._build_geometric_pipeline,
            'color': self._build_color_pipeline,
            'noise': self._build_noise_pipeline
        }
        return builders.get(aug_type, self._build_combined_pipeline)
    
    def _build_lighting_pipeline(self, intensity: float) -> A.Compose:
        """🌟 Pipeline variasi pencahayaan"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.25 * intensity,
                contrast_limit=0.2 * intensity,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.6),
            A.CLAHE(clip_limit=2.0 * intensity, tile_grid_size=(8, 8), p=0.4),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.3
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_position_pipeline(self, intensity: float) -> A.Compose:
        """📍 Pipeline variasi posisi"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=int(15 * intensity), p=0.8),
            A.Affine(
                translate_percent={
                    'x': (-0.1 * intensity, 0.1 * intensity),
                    'y': (-0.1 * intensity, 0.1 * intensity)
                },
                scale=(1 - 0.05 * intensity, 1 + 0.05 * intensity),
                rotate=(-12 * intensity, 12 * intensity),
                p=0.7
            ),
            A.Perspective(scale=(0.02 * intensity, 0.08 * intensity), p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_combined_pipeline(self, intensity: float) -> A.Compose:
        """🎯 Pipeline kombinasi (default untuk research)"""
        return A.Compose([
            # Position variations
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=int(12 * intensity), p=0.7),
            A.Affine(
                translate_percent={
                    'x': (-0.08 * intensity, 0.08 * intensity),
                    'y': (-0.08 * intensity, 0.08 * intensity)
                },
                scale=(1 - 0.04 * intensity, 1 + 0.04 * intensity),
                rotate=(-10 * intensity, 10 * intensity),
                p=0.6
            ),
            A.Perspective(scale=(0.02 * intensity, 0.06 * intensity), p=0.3),
            
            # Lighting variations
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * intensity,
                contrast_limit=0.15 * intensity,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.5),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 1),
                shadow_dimension=4,
                p=0.2
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_geometric_pipeline(self, intensity: float) -> A.Compose:
        """🔄 Pipeline transformasi geometri"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=int(20 * intensity), p=0.7),
            A.Affine(
                translate_percent={
                    'x': (-0.15 * intensity, 0.15 * intensity),
                    'y': (-0.15 * intensity, 0.15 * intensity)
                },
                scale=(1 - 0.1 * intensity, 1 + 0.1 * intensity),
                rotate=(-15 * intensity, 15 * intensity),
                p=0.7
            ),
            A.Perspective(scale=(0.05 * intensity, 0.1 * intensity), p=0.3)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_color_pipeline(self, intensity: float) -> A.Compose:
        """🎨 Pipeline variasi warna"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3 * intensity,
                contrast_limit=0.3 * intensity,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(20 * intensity),
                sat_shift_limit=int(30 * intensity),
                val_shift_limit=int(20 * intensity),
                p=0.6
            ),
            A.CLAHE(clip_limit=3.0 * intensity, tile_grid_size=(8, 8), p=0.5),
            A.ColorJitter(
                brightness=0.2 * intensity,
                contrast=0.2 * intensity,
                saturation=0.2 * intensity,
                hue=0.1 * intensity,
                p=0.5
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_noise_pipeline(self, intensity: float) -> A.Compose:
        """📡 Pipeline noise dan blur"""
        return A.Compose([
            A.GaussNoise(var_limit=(10 * intensity, 50 * intensity), p=0.5),
            A.ISONoise(
                color_shift=(0.01 * intensity, 0.05 * intensity),
                intensity=(0.1 * intensity, 0.5 * intensity),
                p=0.3
            ),
            A.Blur(blur_limit=int(3 * intensity), p=0.3),
            A.MotionBlur(blur_limit=int(7 * intensity), p=0.3)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Utility functions
def create_pipeline_factory(config: Dict[str, Any]) -> PipelineFactory:
    """🏭 Factory untuk create pipeline factory"""
    return PipelineFactory(config)

def get_default_augmentation_types() -> list:
    """📋 Get default research augmentation types"""
    return PipelineFactory.DEFAULT_TYPES

def create_augmentation_pipeline(config: Dict[str, Any], aug_type: str = 'combined', 
                               intensity: float = 0.7) -> A.Compose:
    """🚀 One-liner untuk create augmentation pipeline"""
    factory = create_pipeline_factory(config)
    return factory.create_pipeline(aug_type, intensity)