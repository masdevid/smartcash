"""
SmartCash Model Training Package

This package provides the training pipeline and utilities for training SmartCash models.

Key components:
- TrainingPipeline: Main pipeline for model training
- DataLoaderFactory: Creates data loaders for training and validation
- OptimizerFactory: Creates optimizers and learning rate schedulers
- LossManager: Manages loss computation and tracking
- EarlyStopping: Implements various early stopping strategies
"""
from typing import Dict, Any, Optional, Callable

# Core training components
from . import layers
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

# Main training API
def start_training(api: 'SmartCashModelAPI', 
                 config: Dict[str, Any], 
                 epochs: int = 10,
                 ui_components: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Start training a SmartCash model.
    
    Args:
        api: Initialized SmartCashModelAPI instance
        config: Training configuration dictionary
        epochs: Total number of training epochs
        ui_components: Dictionary of UI components for progress tracking
        
    Returns:
        Dictionary with training results
    """
    # Create and configure the training pipeline
    pipeline = TrainingPipeline()
    
    # Set up progress tracking if UI components are provided
    if ui_components and 'progress_callback' in ui_components:
        pipeline.set_progress_callback(ui_components['progress_callback'])
    
    # Start training
    return pipeline.run_training(
        epochs=epochs,
        config=config
    )

def get_training_info(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get training configuration and dataset information.
    
    Args:
        config: Optional configuration dictionary. If None, default config is used.
        
    Returns:
        Dictionary containing training configuration and dataset information
    """
    from .training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    return pipeline.get_training_info(config or {})

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