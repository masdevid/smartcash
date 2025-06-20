"""
File: smartcash/model/__init__.py
Deskripsi: Model package main exports dengan Fase 1 + Fase 2 integration
"""

# Fase 1: Core Model API
from .api.core import SmartCashModelAPI, create_model_api
from .core.model_builder import ModelBuilder, SmartCashYOLO
from .core.checkpoint_manager import CheckpointManager
from .utils.backbone_factory import BackboneFactory, create_backbone
from .utils.device_utils import setup_device, get_device_info
from .utils.progress_bridge import ModelProgressBridge

# Fase 2: Training Pipeline
from .training import (
    TrainingService, start_training, resume_training,
    create_training_service, get_training_info
)

# Quick start functions
def quick_build_model(backbone='efficientnet_b4', config=None):
    """Quick model building untuk development"""
    api = create_model_api(config)
    result = api.build_model(backbone=backbone)
    return api if result.get('success') else None

def quick_train_model(backbone='efficientnet_b4', epochs=10, config=None, ui_components=None):
    """Quick training pipeline setup"""
    api = quick_build_model(backbone, config)
    if api:
        return start_training(api, config, epochs, ui_components)
    return {'success': False, 'error': 'Failed to build model'}

def get_model_status():
    """Get comprehensive model package status"""
    device_info = get_device_info()
    training_info = get_training_info()
    
    return {
        'fase_1_ready': True,  # Core API
        'fase_2_ready': True,  # Training pipeline
        'device': device_info,
        'training': training_info,
        'quick_functions': ['quick_build_model', 'quick_train_model']
    }

# Export semua untuk backward compatibility
__all__ = [
    # Fase 1 - Core API
    'SmartCashModelAPI', 'create_model_api',
    'ModelBuilder', 'SmartCashYOLO', 
    'CheckpointManager',
    'BackboneFactory', 'create_backbone',
    'setup_device', 'get_device_info',
    'ModelProgressBridge',
    
    # Fase 2 - Training
    'TrainingService', 'start_training', 'resume_training',
    'create_training_service', 'get_training_info',
    
    # Quick functions
    'quick_build_model', 'quick_train_model', 'get_model_status'
]