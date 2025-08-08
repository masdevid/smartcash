"""
File: smartcash/model/training/__init__.py
Description: Training package exports for the SmartCash model training pipeline
"""

# Core training components
from . import layers  # New layers package
from .data_loader_factory import DataLoaderFactory, create_data_loaders, get_dataset_stats
from .utils.metrics_history import create_metrics_recorder
from .optimizer_factory import (
    OptimizerFactory, WarmupScheduler, 
    create_optimizer_and_scheduler, setup_optimizer_with_warmup,
    get_parameter_count
)
from .loss_manager import LossManager, create_loss_manager, compute_yolo_loss
from .utils.progress_tracker import (
    TrainingProgressTracker, create_training_progress_bridge,
    create_simple_progress_callback
)
from .early_stopping import (
    EarlyStopping, StandardEarlyStopping, MultiMetricEarlyStopping, 
    AdaptiveEarlyStopping, PhaseSpecificEarlyStopping,
    create_early_stopping, create_adaptive_early_stopping, create_phase_specific_early_stopping
)
from .platform_presets import PlatformPresets, get_platform_presets, get_platform_config
from .training_pipeline import TrainingPipeline
# run_full_training_pipeline moved to smartcash.model.api.core

# Main training API - now uses merged training pipeline
def start_training(backbone='cspdarknet', phase_1_epochs=1, phase_2_epochs=1, 
                  progress_callback=None, **kwargs):
    """
    Quick start training using merged training pipeline
    
    Args:
        backbone: Model backbone ('cspdarknet' or 'efficientnet_b4')
        phase_1_epochs: Number of epochs for phase 1
        phase_2_epochs: Number of epochs for phase 2
        progress_callback: Progress callback function
        **kwargs: Additional configuration overrides
        
    Returns:
        Training results
    """
    from smartcash.model.api.core import run_full_training_pipeline
    return run_full_training_pipeline(
        backbone=backbone,
        phase_1_epochs=phase_1_epochs,
        phase_2_epochs=phase_2_epochs,
        progress_callback=progress_callback,
        **kwargs
    )

def get_training_info(config=None):
    """Get training configuration dan dataset info"""
    data_factory = DataLoaderFactory(config)
    dataset_info = data_factory.get_dataset_info()
    
    training_config = config or {}
    
    return {
        'dataset_info': dataset_info,
        'training_config': training_config.get('training', {}),
        'available_optimizers': ['adam', 'adamw', 'sgd', 'rmsprop'],
        'available_schedulers': ['cosine', 'step', 'plateau', 'exponential', 'multistep', 'cyclic'],
        'device_info': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    }

# Export main components
__all__ = [
    # Main training API
    'start_training', 'get_training_info',
    
    # Layer management
    'layers',
    
    # Data loading
    'DataLoaderFactory', 'create_data_loaders', 'get_dataset_stats',
    
    # Metrics
    'create_metrics_recorder',
    
    # Optimization
    'OptimizerFactory', 'WarmupScheduler', 'create_optimizer_and_scheduler',
    'setup_optimizer_with_warmup', 'get_parameter_count',
    
    # Loss calculation
    'LossManager', 'create_loss_manager', 'compute_yolo_loss',
    
    # Progress tracking
    'TrainingProgressTracker', 'create_training_progress_bridge',
    'create_simple_progress_callback',
    
    # Early stopping
    'EarlyStopping', 'StandardEarlyStopping', 'MultiMetricEarlyStopping', 
    'AdaptiveEarlyStopping', 'PhaseSpecificEarlyStopping',
    'create_early_stopping', 'create_adaptive_early_stopping', 'create_phase_specific_early_stopping',
    
    # Platform-aware training
    'PlatformPresets', 'get_platform_presets', 'get_platform_config',
    'TrainingPipeline'
]