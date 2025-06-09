"""
File: smartcash/dataset/augmentor/__init__.py
Deskripsi: Main augmentation module dengan sampling service
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

# Main API functions

def create_augmentor(config=None, progress_tracker=None):
    """🚀 Factory dengan config validation"""
    from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config
    
    if config is None:
        validated_config = get_default_augmentation_config()
    else:
        validated_config = validate_augmentation_config(config)
    
    return AugmentationService(validated_config, progress_tracker)

def augment_and_normalize(config=None, target_split='train', progress_tracker=None, progress_callback=None):
    """🎯 One-liner dengan config validation"""
    service = create_augmentor(config, progress_tracker)
    return service.run_augmentation_pipeline(target_split, progress_callback)

def get_sampling_data(config=None, target_split='train', max_samples=5, progress_tracker=None):
    """📊 NEW: Get sampling data untuk evaluasi"""
    service = create_augmentor(config, progress_tracker)
    return service.get_sampling(target_split, max_samples)

def get_supported_types():
    """📋 Get supported augmentation types"""
    return get_default_augmentation_types()

def cleanup_augmented_data(config, target_split=None, progress_tracker=None):
    """🧹 One-liner untuk cleanup augmented data"""
    service = create_augmentor(config, progress_tracker)
    return service.cleanup_augmented_data(target_split)

def get_augmentation_status(config, progress_tracker=None):
    """📊 One-liner untuk get augmentation status"""
    service = create_augmentor(config, progress_tracker)
    return service.get_augmentation_status()

# Export semua untuk backward compatibility
__all__ = [
    # Main classes
    'AugmentationService',
    'AugmentationEngine', 
    'NormalizationEngine',
    'PipelineFactory',
    'ProgressBridge',
    
    # Factory functions
    'create_augmentation_service',
    'create_augmentation_engine',
    'create_normalization_engine',
    'create_pipeline_factory',
    'create_progress_bridge',
    
    # Main API
    'create_augmentor',
    'augment_and_normalize',
    'get_sampling_data',
    'get_supported_types',
    'cleanup_augmented_data',
    'get_augmentation_status',
    
    # Utility functions
    'run_augmentation_pipeline',
    'get_augmentation_samples',
    'augment_dataset_split',
    'normalize_augmented_dataset',
    'get_default_augmentation_types',
    'create_augmentation_pipeline',
    'make_progress_callback'
]