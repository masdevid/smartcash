"""
File: smartcash/ui/model/train/constants.py
Constants for train module following UI module structure standard.
"""

from enum import Enum
from typing import Dict, List, Any

# ==================== Enums ====================

class TrainingOperation(Enum):
    """Operations available in training module."""
    START = "start"
    STOP = "stop"
    RESUME = "resume"
    VALIDATE = "validate"

class TrainingPhase(Enum):
    """Training phases enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"

class LayerMode(Enum):
    """Training layer modes - continuing from backbone module."""
    SINGLE = "single"
    MULTILAYER = "multilayer"

class OptimizationType(Enum):
    """Optimization types for model naming."""
    DEFAULT = "default"
    OPTIMIZED = "optimized"
    ADVANCED = "advanced"

# ==================== Progress Steps ====================

PROGRESS_STEPS = {
    TrainingOperation.START.value: [
        "🔄 Loading backbone configuration",
        "🏗️ Building model from backbone", 
        "📊 Preparing training data",
        "⚙️ Setting up optimizer and scheduler",
        "🚀 Starting training loop",
        "📈 Training in progress"
    ],
    TrainingOperation.STOP.value: [
        "🛑 Stopping current training",
        "💾 Saving best model",
        "🧹 Cleanup and finalization"
    ],
    TrainingOperation.RESUME.value: [
        "📂 Loading checkpoint",
        "🔄 Restoring training state",
        "✅ Validating loaded model",
        "🚀 Resuming training loop"
    ],
    TrainingOperation.VALIDATE.value: [
        "📊 Loading validation data",
        "🔍 Running model validation",
        "📈 Computing metrics",
        "✅ Validation complete"
    ]
}

# ==================== UI Configuration ====================

UI_CONFIG = {
    'module_name': 'train',
    'parent_module': 'model',
    'title': '🚀 Model Training',
    'subtitle': 'Train currency detection models with real-time monitoring',
    'description': 'Continue training process from backbone configuration with live charts',
    'icon': '🚀',
    'version': '2.0.0'
}

# Module Metadata
MODULE_METADATA = {
    'name': 'train',
    'title': '🚀 Model Training',
    'description': 'Model training module with live monitoring and progress tracking',
    'version': '2.0.0',
    'category': 'model',
    'author': 'SmartCash',
    'tags': ['training', 'model', 'monitoring', 'charts', 'progress'],
    'features': [
        'Single/multilayer training options',
        'Live loss and mAP charts',
        'Real-time progress tracking',
        'Best model automatic saving',
        'Backend training service integration',
        'Backbone configuration continuation'
    ]
}

# Button Configuration
BUTTON_CONFIG = {
    'start': {
        'text': '🚀 Start Training',
        'style': 'primary',
        'tooltip': 'Start training with current configuration',
        'order': 1
    },
    'stop': {
        'text': '🛑 Stop Training',
        'style': 'danger',
        'tooltip': 'Stop current training and save best model',
        'order': 2
    },
    'resume': {
        'text': '🔄 Resume Training',
        'style': 'warning',
        'tooltip': 'Resume training from last checkpoint',
        'order': 3
    },
    'validate': {
        'text': '📊 Validate Model',
        'style': 'info',
        'tooltip': 'Run validation on current best model',
        'order': 4
    }
}

# ==================== Training Configuration ====================

DEFAULT_TRAINING_CONFIG = {
    'layer_mode': 'single',  # single or multilayer
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.001,
    'validation_interval': 1,
    'save_interval': 5,
    'optimization_type': 'default',
    'early_stopping': {
        'enabled': True,
        'patience': 15,
        'metric': 'val_map50',
        'mode': 'max'
    },
    'mixed_precision': True,
    'gradient_accumulation': 1
}

DEFAULT_OPTIMIZER_CONFIG = {
    'type': 'adam',
    'weight_decay': 0.0005,
    'momentum': 0.9
}

DEFAULT_SCHEDULER_CONFIG = {
    'type': 'cosine',
    'warmup_epochs': 5,
    'min_lr': 0.00001
}

# ==================== Layer Configuration ====================

LAYER_CONFIGS = {
    LayerMode.SINGLE.value: {
        'display_name': 'Single Layer',
        'description': 'Single detection layer (banknote only)',
        'layers': ['banknote'],
        'classes': 1,
        'recommended_epochs': 100,
        'recommended_lr': 0.001
    },
    LayerMode.MULTILAYER.value: {
        'display_name': 'Multi-Layer',
        'description': 'Multiple detection layers (banknote + nominal + security)',
        'layers': ['banknote', 'nominal', 'security'],
        'classes': 3,
        'recommended_epochs': 150,
        'recommended_lr': 0.0008
    }
}

# ==================== Model Naming Convention ====================

MODEL_NAMING_TEMPLATE = "{backbone}_{layer}_{optimization}"

def generate_model_name(backbone: str, layer_mode: str, optimization_type: str = "default") -> str:
    """
    Generate model name following the naming convention.
    
    Args:
        backbone: Backbone type (e.g., 'efficientnet_b4', 'cspdarknet')
        layer_mode: Layer mode ('single', 'multilayer')
        optimization_type: Optimization type ('default', 'optimized', 'advanced')
    
    Returns:
        Model name string
    """
    return MODEL_NAMING_TEMPLATE.format(
        backbone=backbone,
        layer=layer_mode,
        optimization=optimization_type
    )

# ==================== Chart Configuration ====================

CHART_CONFIG = {
    'loss_chart': {
        'title': 'Training Loss',
        'y_label': 'Loss Value',
        'color': '#ff6b6b',
        'metrics': ['train_loss', 'val_loss'],
        'update_interval': 1000,  # milliseconds
        'max_points': 1000
    },
    'map_chart': {
        'title': 'mAP Performance',
        'y_label': 'mAP Score',
        'color': '#4ecdc4',
        'metrics': ['val_map50', 'val_map75'],
        'update_interval': 1000,  # milliseconds
        'max_points': 1000
    }
}

# ==================== Status Indicators ====================

STATUS_INDICATORS = {
    TrainingPhase.IDLE.value: {
        'icon': '⏸️',
        'color': 'gray',
        'message': 'Ready to start training'
    },
    TrainingPhase.INITIALIZING.value: {
        'icon': '🔄',
        'color': 'blue',
        'message': 'Initializing training components'
    },
    TrainingPhase.TRAINING.value: {
        'icon': '🚀',
        'color': 'green',
        'message': 'Training in progress'
    },
    TrainingPhase.VALIDATING.value: {
        'icon': '📊',
        'color': 'orange',
        'message': 'Running validation'
    },
    TrainingPhase.COMPLETED.value: {
        'icon': '✅',
        'color': 'success',
        'message': 'Training completed successfully'
    },
    TrainingPhase.STOPPED.value: {
        'icon': '🛑',
        'color': 'warning',
        'message': 'Training stopped by user'
    },
    TrainingPhase.ERROR.value: {
        'icon': '❌',
        'color': 'danger',
        'message': 'Training error occurred'
    }
}

# ==================== Validation Settings ====================

VALIDATION_CONFIG = {
    'required_fields': ['layer_mode', 'epochs', 'batch_size', 'learning_rate'],
    'epochs': {'min': 1, 'max': 1000},
    'batch_size': {'min': 1, 'max': 256},
    'learning_rate': {'min': 1e-6, 'max': 1.0},
    'validation_interval': {'min': 1, 'max': 50},
    'save_interval': {'min': 1, 'max': 100}
}

# ==================== Model Storage Paths ====================

MODEL_STORAGE = {
    'checkpoints_dir': '/data/checkpoints',
    'best_models_dir': '/data/models/best',
    'logs_dir': '/data/logs/training',
    'tensorboard_dir': '/data/logs/tensorboard'
}

# ==================== Default Messages ====================

ERROR_MESSAGES = {
    'training_failed': "Training process failed",
    'model_load_failed': "Failed to load model from backbone",
    'data_load_failed': "Failed to load training data",
    'config_invalid': "Training configuration is invalid",
    'backbone_missing': "Backbone configuration not found"
}

SUCCESS_MESSAGES = {
    'training_started': "Training started successfully",
    'training_completed': "Training completed successfully",
    'model_saved': "Best model saved successfully",
    'validation_completed': "Validation completed successfully"
}

# ==================== Defaults ====================

DEFAULT_CONFIG = {
    'training': DEFAULT_TRAINING_CONFIG.copy(),
    'optimizer': DEFAULT_OPTIMIZER_CONFIG.copy(),
    'scheduler': DEFAULT_SCHEDULER_CONFIG.copy(),
    'model_storage': MODEL_STORAGE.copy(),
    'ui': {
        'show_advanced_options': False,
        'auto_save_best': True,
        'live_charts_enabled': True,
        'progress_updates_enabled': True
    }
}