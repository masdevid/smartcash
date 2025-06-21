"""
File: smartcash/model/training/__init__.py
Deskripsi: Training package exports untuk Fase 2 implementation
"""

from .training_service import TrainingService, create_training_service, quick_train_model
from .data_loader_factory import DataLoaderFactory, create_data_loaders, get_dataset_stats
from .metrics_tracker import MetricsTracker, create_metrics_tracker
from .optimizer_factory import (
    OptimizerFactory, WarmupScheduler, 
    create_optimizer_and_scheduler, setup_optimizer_with_warmup,
    get_parameter_count
)
from .loss_manager import LossManager, create_loss_manager, compute_yolo_loss
from .utils.training_progress_bridge import (
    TrainingProgressBridge, create_training_progress_bridge,
    create_simple_progress_callback
)
from .utils.early_stopping import (
    EarlyStopping, MultiMetricEarlyStopping, AdaptiveEarlyStopping,
    create_early_stopping, create_adaptive_early_stopping
)

# Main training API
def start_training(model_api, config=None, epochs=100, ui_components=None,
                  progress_callback=None, metrics_callback=None):
    """
    Quick start training dengan default setup
    
    Args:
        model_api: SmartCashModelAPI dari Fase 1
        config: Training configuration (optional)
        epochs: Number of epochs
        ui_components: UI components untuk progress tracking
        progress_callback: Progress callback function
        metrics_callback: Metrics callback function
        
    Returns:
        Training results
    """
    service = create_training_service(
        model_api=model_api,
        config=config, 
        ui_components=ui_components,
        progress_callback=progress_callback,
        metrics_callback=metrics_callback
    )
    
    return service.start_training(epochs=epochs)

def resume_training(model_api, checkpoint_path, additional_epochs=50, 
                   config=None, ui_components=None):
    """
    Resume training dari checkpoint
    
    Args:
        model_api: SmartCashModelAPI instance
        checkpoint_path: Path ke checkpoint file
        additional_epochs: Additional epochs untuk training
        config: Training configuration
        ui_components: UI components
        
    Returns:
        Training results
    """
    service = create_training_service(model_api, config, ui_components)
    return service.resume_training(checkpoint_path, additional_epochs)

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

# Export semua components
__all__ = [
    # Main service
    'TrainingService', 'create_training_service', 'start_training', 'resume_training',
    
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
    'TrainingProgressBridge', 'create_training_progress_bridge',
    'create_simple_progress_callback',
    
    # Early stopping
    'EarlyStopping', 'MultiMetricEarlyStopping', 'AdaptiveEarlyStopping',
    'create_early_stopping', 'create_adaptive_early_stopping',
    
    # Convenience functions
    'quick_train_model', 'get_training_info'
]