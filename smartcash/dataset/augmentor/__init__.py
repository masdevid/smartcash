"""
File: smartcash/dataset/augmentor/__init__.py
Deskripsi: Updated main augmentation module dengan FileNamingManager dan sample generator integration
"""

from smartcash.dataset.augmentor.service import (
    AugmentationService,
    create_augmentation_service,
    run_augmentation_pipeline,
    get_augmentation_samples
)

from smartcash.dataset.augmentor.core.engine import (
    AugmentationEngine,
    create_augmentation_engine,
    augment_dataset_split
)

from smartcash.dataset.augmentor.core.normalizer import (
    NormalizationEngine,
    create_normalization_engine,
    normalize_augmented_dataset
)

from smartcash.dataset.augmentor.core.pipeline_factory import (
    PipelineFactory,
    create_pipeline_factory,
    get_default_augmentation_types,
    create_augmentation_pipeline
)

from smartcash.dataset.augmentor.utils.progress_bridge import (
    ProgressBridge,
    create_progress_bridge,
    make_progress_callback
)

# NEW: Sample generator integration
from smartcash.dataset.augmentor.utils.sample_generator import (
    AugmentationSampleGenerator,
    create_augmentation_sample_generator,
    generate_augmentation_samples,
    cleanup_augmentation_samples
)

# Main API functions

def create_augmentor(config=None, progress_tracker=None):
    """🚀 Factory dengan config validation dan FileNamingManager"""
    from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config
    
    if config is None:
        validated_config = get_default_augmentation_config()
    else:
        validated_config = validate_augmentation_config(config)
    
    return AugmentationService(validated_config, progress_tracker)

def augment_and_normalize(config=None, target_split='train', progress_tracker=None, progress_callback=None):
    """🎯 One-liner dengan FileNamingManager dan preprocessor API integration"""
    service = create_augmentor(config, progress_tracker)
    return service.run_augmentation_pipeline(target_split, progress_callback)

def create_live_preview(config=None, target_split='train', progress_tracker=None):
    """🎥 NEW: Create live preview augmentation tanpa normalization"""
    service = create_augmentor(config, progress_tracker)
    return service.create_live_preview(target_split)

def get_sampling_data(config=None, target_split='train', max_samples=5, progress_tracker=None):
    """📊 NEW: Get sampling data dengan sample_aug_* pattern generation"""
    service = create_augmentor(config, progress_tracker)
    return service.get_sampling(target_split, max_samples)

def get_supported_types():
    """📋 Get supported augmentation types"""
    return get_default_augmentation_types()

# NEW: Configurable cleanup functions
def cleanup_augmented_data(config, target_split=None, progress_tracker=None):
    """🧹 Cleanup augmented files + preprocessed .npy files"""
    service = create_augmentor(config, progress_tracker)
    return service.cleanup_augmented_data(target_split)

def cleanup_samples(config, target_split=None, progress_tracker=None):
    """🧹 Cleanup sample_aug_* files dari preprocessed directory"""
    service = create_augmentor(config, progress_tracker)
    return service.cleanup_samples(target_split)

def cleanup_all(config, target_split=None, progress_tracker=None):
    """🧹 Cleanup semua: augmented + samples"""
    service = create_augmentor(config, progress_tracker)
    return service.cleanup_all(target_split)

def cleanup_data(config, target_split=None, target='both', progress_tracker=None):
    """🧹 Configurable cleanup dengan target selection"""
    service = create_augmentor(config, progress_tracker)
    return service.cleanup_data(target_split, target)

def get_augmentation_status(config, progress_tracker=None):
    """📊 Status dengan FileNamingManager pattern detection"""
    service = create_augmentor(config, progress_tracker)
    return service.get_augmentation_status()

# NEW: Sample generation utilities
def generate_preview_samples(config, target_split='train', max_samples=5, max_per_class=2):
    """📸 Generate preview samples dengan FileNamingManager patterns"""
    generator = create_augmentation_sample_generator(config)
    return generator.generate_augmentation_samples(target_split, max_samples, max_per_class)

def get_sample_statistics(config, target_split='train'):
    """📊 Get statistics dari sample_aug_* files"""
    generator = create_augmentation_sample_generator(config)
    return generator.get_sample_statistics(target_split)

# Export semua untuk backward compatibility
__all__ = [
    # Main classes
    'AugmentationService',
    'AugmentationEngine', 
    'NormalizationEngine',
    'PipelineFactory',
    'ProgressBridge',
    'AugmentationSampleGenerator',  # NEW
    
    # Factory functions
    'create_augmentation_service',
    'create_augmentation_engine',
    'create_normalization_engine',
    'create_pipeline_factory',
    'create_progress_bridge',
    'create_augmentation_sample_generator',  # NEW
    
    # Main API
    'create_augmentor',
    'augment_and_normalize',
    'get_sampling_data',
    'get_supported_types',
    'get_augmentation_status',
    'create_live_preview',  # NEW
    
    # NEW: Cleanup API (configurable)
    'cleanup_augmented_data',
    'cleanup_samples', 
    'cleanup_all',
    'cleanup_data',
    
    # NEW: Sample generation API
    'generate_preview_samples',
    'generate_augmentation_samples',
    'cleanup_augmentation_samples',
    'get_sample_statistics',
    
    # Utility functions
    'run_augmentation_pipeline',
    'get_augmentation_samples',
    'augment_dataset_split',
    'normalize_augmented_dataset',
    'get_default_augmentation_types',
    'create_augmentation_pipeline',
    'make_progress_callback'
]