"""
File: smartcash/model/__init__.py
Deskripsi: Model package main exports dengan Fase 1 + Fase 2 integration
"""

# Fase 1: Core Model API
from .api.core import SmartCashModelAPI, create_model_api
from .core.checkpoint_manager import CheckpointManager
from .utils.backbone_factory import BackboneFactory
from .utils.device_utils import setup_device, get_device_info
from .utils.progress_bridge import ModelProgressBridge

# Initialize backbone factory
backbone_factory = BackboneFactory()

def create_backbone(backbone_type: str, pretrained: bool = True, **kwargs):
    """Create a backbone model using the factory"""
    return backbone_factory.create_backbone(backbone_type, pretrained, **kwargs)

# Fase 2: Training Pipeline
from .training import (
    TrainingService, start_training, resume_training,
    create_training_service, get_training_info
)

# Fase 3: Analysis & Evaluation
from .analysis.analysis_service import AnalysisService
from .analysis.class_analyzer import ClassAnalyzer
from .analysis.currency_analyzer import CurrencyAnalyzer
from .analysis.layer_analyzer import LayerAnalyzer

from .evaluation import (
    EvaluationService, CheckpointSelector, EvaluationMetrics, ScenarioManager
)

# Fase 4: Reporting
from .reporting import (
    ReportService, ResearchGenerator, SummaryGenerator, ComparisonGenerator
)

# Re-export quick functions from API
from .api.core import quick_build_model

def quick_train_model(backbone='efficientnet_b4', epochs=10, config=None, ui_components=None):
    """Quick training pipeline setup"""
    api = quick_build_model(backbone=backbone, config=config)
    if api and hasattr(api, 'build_model'):
        build_result = api.build_model(backbone=backbone)
        if build_result.get('success'):
            return start_training(api, config, epochs, ui_components)
    return {'success': False, 'error': 'Failed to build model'}

def get_model_status():
    """Get comprehensive model package status"""
    return {
        'core_ready': True,      # Fase 1: Core API
        'training_ready': True,  # Fase 2: Training
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
    'ModelProgressBridge',
    
    # Fase 2 - Training
    'TrainingService', 'start_training', 'resume_training',
    'create_training_service', 'get_training_info',
    
    # Fase 3 - Analysis & Evaluation
    'AnalysisService', 'ClassAnalyzer', 'CurrencyAnalyzer', 
    'LayerAnalyzer', 'EvaluationService', 'CheckpointSelector',
    'EvaluationMetrics', 'ScenarioManager',
    
    # Fase 4 - Reporting
    'ReportService', 'ResearchGenerator', 'SummaryGenerator', 
    'ComparisonGenerator',
    
    # Quick functions
    'quick_build_model', 'quick_train_model', 'get_model_status'
]

# Initialize package-level logger
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())