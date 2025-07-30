"""
File: smartcash/model/training/__init__.py
Deskripsi: Training package exports untuk Fase 2 implementation
"""

# Old training service removed - use merged training_pipeline instead
from .data_loader_factory import DataLoaderFactory, create_data_loaders, get_dataset_stats
from .metrics_tracker import MetricsTracker, create_metrics_tracker
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
from .utils.early_stopping import (
    EarlyStopping, MultiMetricEarlyStopping, AdaptiveEarlyStopping,
    create_early_stopping, create_adaptive_early_stopping
)
from .visualization_manager import ComprehensiveMetricsTracker, create_visualization_manager
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
    
    # Data loading
    'DataLoaderFactory', 'create_data_loaders', 'get_dataset_stats',
    
    # Metrics
    'MetricsTracker', 'create_metrics_tracker',
    
    # Optimization
    'OptimizerFactory', 'WarmupScheduler', 'create_optimizer_and_scheduler',
    'setup_optimizer_with_warmup', 'get_parameter_count',
    
    # Loss calculation
    'LossManager', 'create_loss_manager', 'compute_yolo_loss',
    
    # Progress tracking
    'TrainingProgressTracker', 'create_training_progress_bridge',
    'create_simple_progress_callback',
    
    # Early stopping
    'EarlyStopping', 'MultiMetricEarlyStopping', 'AdaptiveEarlyStopping',
    'create_early_stopping', 'create_adaptive_early_stopping',
    
    # Visualization
    'ComprehensiveMetricsTracker', 'create_visualization_manager',
    
    # Platform-aware training
    'PlatformPresets', 'get_platform_presets', 'get_platform_config',
    'TrainingPipeline'
]