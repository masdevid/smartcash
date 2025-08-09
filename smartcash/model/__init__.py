"""
SmartCash Model Package

This package provides the core functionality for the SmartCash object detection models,
including model creation, training, and evaluation.

Main components:
- api: Core model API and training pipeline
- training: Training utilities and pipeline
- evaluation: Model evaluation and analysis
- architectures: Model architectures and components
"""

from typing import Dict, Any, Optional

# Core Model API
from .api import SmartCashModelAPI, create_api
from .core.checkpoints.checkpoint_manager import CheckpointManager
from .utils.backbone_factory import BackboneFactory
from .utils.device_utils import setup_device, get_device_info
from .training.utils.progress_tracker import TrainingProgressTracker

# Initialize backbone factory
backbone_factory = BackboneFactory()

def create_backbone(backbone_type: str, pretrained: bool = False, **kwargs):
    """Create a backbone model using the factory.
    
    Args:
        backbone_type: Type of backbone to create (e.g., 'efficientnet_b4', 'yolov5s')
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for the backbone
        
    Returns:
        Initialized backbone model
    """
    return backbone_factory.create_backbone(backbone_type, pretrained, **kwargs)

# Training Pipeline
from .training import start_training, get_training_info

# Analysis & Evaluation
from .evaluation import EvaluationService, CheckpointSelector, ScenarioManager

def quick_train_model(backbone: str = 'efficientnet_b4', 
                    epochs: int = 10, 
                    config: Optional[Dict[str, Any]] = None, 
                    ui_components: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quick training pipeline setup.
    
    Args:
        backbone: Backbone architecture to use
        epochs: Number of training epochs
        config: Training configuration
        ui_components: UI components for progress tracking
        
    Returns:
        Dictionary with training results
    """
    # Create API instance with default config
    api = create_api(config=config or {})
    
    # Build model with specified backbone
    build_result = api.build_model(backbone=backbone)
    if not build_result.get('success'):
        return {'success': False, 'error': 'Failed to build model'}
        
    # Start training
    return start_training(api, config or {}, epochs, ui_components)

def get_model_status() -> Dict[str, bool]:
    """Get comprehensive model package status.
    
    Returns:
        Dictionary indicating which components are ready
    """
    return {
        'core_ready': True,      # Core API
        'training_ready': True,  # Training pipeline
        'analysis_ready': True,  # Fase 3: Analysis
        'evaluation_ready': True,# Fase 3: Evaluation
        'reporting_ready': True, # Fase 4: Reporting
        'device': get_device_info(),
        'training': get_training_info(),
        'quick_functions': ['quick_build_model', 'quick_train_model']
    }

# Export semua untuk backward compatibility
__all__ = [
    # Fase 1 - Core API
    'SmartCashModelAPI', 'create_model_api',
    'CheckpointManager', 'create_backbone',
    'setup_device', 'get_device_info',
    'TrainingProgressTracker',
    
    # Fase 2 - Training
    'TrainingService', 'start_training', 'resume_training',
    'create_training_service', 'get_training_info',
    
    # Fase 3 - Analysis & Evaluation
    'EvaluationService', 'CheckpointSelector',
    'EvaluationMetrics', 'ScenarioManager',
    
    # Fase 4 - Reporting
    
    # Quick functions
    'quick_build_model', 'quick_train_model', 'get_model_status'
]

# Initialize package-level logger
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())