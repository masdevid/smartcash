"""
File: smartcash/dataset/augmentor/core/pipeline_factory.py
Deskripsi: Factory untuk augmentation pipelines dengan config structure yang diusulkan
"""

import albumentations as A
from typing import Dict, Any, Callable

class PipelineFactory:
    """🏭 Factory untuk augmentation pipelines dengan proposed config structure"""
    
    DEFAULT_TYPES = ['lighting', 'position', 'combined']
    
    def __init__(self, config: Dict[str, Any] = None):
        from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config
        
        if config is None:
            from smartcash.dataset.augmentor.utils.config_validator import get_default_augmentation_config
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        self.aug_config = self.config.get('augmentation', {})
    
    def create_pipeline(self, aug_type: str = None, intensity: float = None) -> A.Compose:
        """Create pipeline dengan config defaults"""
        aug_type = aug_type or self.aug_config.get('types', ['combined'])[0]
        intensity = intensity if intensity is not None else self.aug_config.get('intensity', 0.7)
        
        # Validate intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Use custom pipeline untuk combined type
        if aug_type == 'combined':
            return self.create_custom_pipeline(self.aug_config)
        
        # Get pipeline builder untuk other types
        builder = self._get_pipeline_builder(aug_type)
        return builder(intensity)
    
    def create_custom_pipeline(self, aug_config: Dict[str, Any]) -> A.Compose:
        """🎯 Create pipeline dengan config structure yang diusulkan"""
        # Extract dari combined section (struktur yang diusulkan)
        combined_config = aug_config.get('combined', {})
        
        horizontal_flip = combined_config.get('horizontal_flip', 0.5)
        rotation_limit = combined_config.get('rotation_limit', 12)
        scale_limit = combined_config.get('scale_limit', 0.04)
        translate_limit = combined_config.get('translate_limit', 0.08)
        brightness_limit = combined_config.get('brightness_limit', 0.2)
        contrast_limit = combined_config.get('contrast_limit', 0.15)
        hsv_hue = combined_config.get('hsv_hue', 10)
        hsv_saturation = combined_config.get('hsv_saturation', 15)
        
        return A.Compose([
            # Position variations
            A.HorizontalFlip(p=horizontal_flip),
            A.Rotate(limit=rotation_limit, p=0.7),
            A.Affine(
                translate_percent={
                    'x': (-translate_limit, translate_limit),
                    'y': (-translate_limit, translate_limit)
                },
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                p=0.6
            ),
            # BANKNOTE SCALE CURRICULUM: Teach model about banknote sizes
            A.OneOf([
                # Large scale: Simulate close-up banknotes (80-95% of image)
                A.Affine(scale=(1.1, 1.3), p=1.0),
                # Medium scale: Standard detection size (50-75% of image) 
                A.Affine(scale=(0.9, 1.1), p=1.0),
                # Small scale: Far banknotes (30-50% of image)
                A.Affine(scale=(0.7, 0.9), p=1.0),
            ], p=0.8),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            
            # Lighting variations
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.5),
            
            # HSV variations
            A.HueSaturationValue(
                hue_shift_limit=hsv_hue,
                sat_shift_limit=hsv_saturation,
                val_shift_limit=10,
                p=0.6
            ),
            
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 1),
                shadow_dimension=4,
                p=0.2
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
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
        """🌟 Pipeline variasi pencahayaan - BALANCED lighting variations (FIXED: reduced extreme values)"""
        lighting_config = self.aug_config.get('lighting', {})
        
        # BALANCED brightness ranges - prevent complete black/white images
        brightness_limit = lighting_config.get('brightness_limit', 0.3) * intensity  # Reduced from 0.6
        contrast_limit = lighting_config.get('contrast_limit', 0.25) * intensity  # Reduced from 0.3
        hsv_hue = lighting_config.get('hsv_hue', 12)  # Reduced from 15
        hsv_saturation = lighting_config.get('hsv_saturation', 15)  # Reduced from 20
        
        # FIXED: Create lighting pipeline with only pixel-level transforms (no bbox processing needed)
        # This eliminates the bbox processor warning for pure lighting augmentation
        return A.Compose([
            # BALANCED lighting variations - prevent extreme black/white images
            A.OneOf([
                # Subtle dark variations (FIXED: was -0.8 to -0.5, now more reasonable)
                A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(-0.2, 0.1), p=1.0),
                # Subtle bright variations (FIXED: was 0.5 to 1.5, now more reasonable)  
                A.RandomBrightnessContrast(brightness_limit=(0.15, 0.4), contrast_limit=(-0.1, 0.2), p=1.0),
                # Standard brightness/contrast variations
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0
                )
            ], p=0.8),
            
            # Moderate gamma correction 
            A.RandomGamma(gamma_limit=(85, 115), p=0.6),
            
            # Moderate HSV variations
            A.HueSaturationValue(
                hue_shift_limit=hsv_hue,
                sat_shift_limit=hsv_saturation,
                val_shift_limit=12,
                p=0.7
            ),
            
            # Subtle lighting enhancement (FIXED: clip_limit must be >= 1.0)
            A.CLAHE(clip_limit=max(1.0, 2.0 * intensity), tile_grid_size=(8, 8), p=0.2)
            
            # NOTE: Removed RandomShadow from lighting pipeline to eliminate bbox processor warning
            # Shadow effects are better handled in combined/position pipelines where bbox processing is expected
        ])
    
    def _build_position_pipeline(self, intensity: float) -> A.Compose:
        """📍 Pipeline variasi posisi"""
        position_config = self.aug_config.get('position', {})
        
        horizontal_flip = position_config.get('horizontal_flip', 0.5)
        rotation_limit = position_config.get('rotation_limit', 12) * intensity
        translate_limit = position_config.get('translate_limit', 0.08) * intensity
        scale_limit = position_config.get('scale_limit', 0.04) * intensity
        
        return A.Compose([
            A.HorizontalFlip(p=horizontal_flip),
            A.Rotate(limit=int(rotation_limit), p=0.8),
            A.Affine(
                translate_percent={
                    'x': (-translate_limit, translate_limit),
                    'y': (-translate_limit, translate_limit)
                },
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                p=0.7
            ),
            A.Perspective(scale=(0.02 * intensity, 0.08 * intensity), p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _build_combined_pipeline(self, intensity: float) -> A.Compose:
        """🎯 Pipeline kombinasi (fallback untuk legacy)"""
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
        ])  # FIXED: Removed bbox_params - color transforms don't affect bounding boxes
    
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
        ])  # FIXED: Removed bbox_params - noise/blur transforms don't affect bounding boxes


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