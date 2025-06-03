"""
File: smartcash/ui/training/constants/training_config_constants.py
Deskripsi: Konstanta untuk konfigurasi training dan model yang digunakan UI
"""

from typing import Dict, Any, List, NamedTuple
from enum import Enum

# ============================================================================
# TRAINING CONFIGURATION SCHEMAS
# ============================================================================

class ConfigSource(Enum):
    """Sumber konfigurasi training"""
    YAML_FILE = "yaml_file"
    DEFAULT = "default"
    USER_INPUT = "user_input"
    MODEL_MANAGER = "model_manager"
    RUNTIME = "runtime"

class ConfigCategory(Enum):
    """Kategori konfigurasi untuk organizing"""
    MODEL = "model"
    TRAINING = "training"
    HYPERPARAMETERS = "hyperparameters"
    BACKBONE = "backbone"
    DETECTOR = "detector"
    TRAINING_STRATEGY = "training_strategy"
    PATHS = "paths"
    OPTIMIZATION = "optimization"

# ============================================================================
# YAML CONFIG FILE MAPPINGS
# ============================================================================

# Mapping YAML files ke config categories
YAML_CONFIG_MAPPING = {
    'model_config.yaml': ConfigCategory.MODEL,
    'training_config.yaml': ConfigCategory.TRAINING,
    'hyperparameter_config.yaml': ConfigCategory.HYPERPARAMETERS,
    'backbone_config.yaml': ConfigCategory.BACKBONE,
    'detector_config.yaml': ConfigCategory.DETECTOR,
    'training_strategy_config.yaml': ConfigCategory.TRAINING_STRATEGY,
    'paths_config.yaml': ConfigCategory.PATHS
}

# Required keys untuk setiap config category
REQUIRED_CONFIG_KEYS = {
    ConfigCategory.MODEL: [
        'model_type', 'backbone', 'num_classes', 'detection_layers'
    ],
    ConfigCategory.TRAINING: [
        'epochs', 'batch_size', 'early_stopping', 'patience'
    ],
    ConfigCategory.HYPERPARAMETERS: [
        'learning_rate', 'weight_decay', 'optimizer'
    ],
    ConfigCategory.BACKBONE: [
        'type', 'pretrained', 'freeze'
    ],
    ConfigCategory.PATHS: [
        'data_dir', 'checkpoint_dir'
    ]
}

# Optional keys dengan default values
OPTIONAL_CONFIG_KEYS = {
    ConfigCategory.MODEL: {
        'layer_mode': 'single',
        'image_size': 640,
        'transfer_learning': True
    },
    ConfigCategory.TRAINING: {
        'save_best': True,
        'save_interval': 10,
        'validation_split': 0.2
    },
    ConfigCategory.HYPERPARAMETERS: {
        'momentum': 0.937,
        'warmup_epochs': 3,
        'scheduler_type': 'cosine'
    },
    ConfigCategory.TRAINING_STRATEGY: {
        'multi_scale': False,
        'mosaic': True,
        'mixup': False
    }
}

# ============================================================================
# MODEL TYPE CONFIGURATIONS
# ============================================================================

class ModelTypeConfig(NamedTuple):
    """Configuration untuk model types"""
    backbone: str
    description: str
    recommended_batch_size: int
    recommended_lr: float
    min_memory_gb: int
    supported_layers: List[str]

# Supported model types (sesuai requirement: efficientnet_b4 dan csp_darknet)
MODEL_TYPE_CONFIGURATIONS = {
    'efficient_optimized': ModelTypeConfig(
        backbone='efficientnet_b4',
        description='EfficientNet-B4 dengan optimasi untuk currency detection',
        recommended_batch_size=16,
        recommended_lr=0.001,
        min_memory_gb=8,
        supported_layers=['banknote', 'nominal', 'security']
    ),
    'yolov5s': ModelTypeConfig(
        backbone='cspdarknet_s',
        description='YOLOv5s baseline dengan CSPDarknet backbone',
        recommended_batch_size=32,
        recommended_lr=0.01,
        min_memory_gb=4,
        supported_layers=['banknote', 'nominal', 'security']
    )
}

# Backbone specific configurations
BACKBONE_CONFIGURATIONS = {
    'efficientnet_b4': {
        'pretrained_url': 'timm://efficientnet_b4',
        'input_size': 640,
        'feature_channels': [56, 160, 448],  # P3, P4, P5
        'stride': 32,
        'memory_efficient': True,
        'supports_attention': True
    },
    'cspdarknet_s': {
        'pretrained_url': 'yolov5s.pt',
        'input_size': 640,
        'feature_channels': [128, 256, 512],  # P3, P4, P5
        'stride': 32,
        'memory_efficient': False,
        'supports_attention': False
    }
}

# ============================================================================
# TRAINING PARAMETER CONSTRAINTS
# ============================================================================

class ParameterConstraint(NamedTuple):
    """Constraint untuk training parameters"""
    min_value: float
    max_value: float
    step: float
    recommended: float
    data_type: type

# Parameter constraints untuk validation
TRAINING_PARAMETER_CONSTRAINTS = {
    'epochs': ParameterConstraint(1, 1000, 1, 100, int),
    'batch_size': ParameterConstraint(1, 128, 1, 16, int),
    'learning_rate': ParameterConstraint(1e-6, 1.0, 1e-6, 0.001, float),
    'weight_decay': ParameterConstraint(0.0, 1.0, 1e-5, 0.0005, float),
    'patience': ParameterConstraint(1, 100, 1, 10, int),
    'save_interval': ParameterConstraint(1, 100, 1, 10, int),
    'image_size': ParameterConstraint(320, 1280, 32, 640, int),
    'momentum': ParameterConstraint(0.0, 1.0, 0.01, 0.937, float)
}

# Dependent parameter relationships
PARAMETER_DEPENDENCIES = {
    'batch_size': {
        'affects': ['memory_usage', 'gradient_stability'],
        'scales_with': ['num_gpus', 'available_memory']
    },
    'learning_rate': {
        'affects': ['convergence_speed', 'final_accuracy'],
        'scales_with': ['batch_size', 'model_size']
    },
    'image_size': {
        'affects': ['memory_usage', 'detection_accuracy'],
        'constrains': ['batch_size']
    }
}

# ============================================================================
# DETECTION LAYER CONFIGURATIONS
# ============================================================================

# Detection layers sesuai dengan SmartCash requirements
DETECTION_LAYER_CONFIGS = {
    'banknote': {
        'num_classes': 7,  # 7 denominasi
        'class_names': ['001', '002', '005', '010', '020', '050', '100'],
        'description': 'Deteksi uang kertas utuh',
        'confidence_threshold': 0.25,
        'nms_threshold': 0.45,
        'priority': 1
    },
    'nominal': {
        'num_classes': 7,  # 7 area nominal
        'class_names': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
        'description': 'Deteksi area nominal pada uang',
        'confidence_threshold': 0.30,
        'nms_threshold': 0.45,
        'priority': 2
    },
    'security': {
        'num_classes': 3,  # 3 fitur keamanan
        'class_names': ['l3_sign', 'l3_text', 'l3_thread'],
        'description': 'Deteksi fitur keamanan',
        'confidence_threshold': 0.35,
        'nms_threshold': 0.45,
        'priority': 3
    }
}

# Layer mode configurations
LAYER_MODE_CONFIGS = {
    'single': {
        'description': 'Single layer detection (recommended untuk training awal)',
        'max_layers': 1,
        'default_layer': 'banknote',
        'memory_multiplier': 1.0,
        'training_complexity': 'low'
    },
    'multilayer': {
        'description': 'Multi-layer detection (advanced)',
        'max_layers': 3,
        'default_layers': ['banknote', 'nominal'],
        'memory_multiplier': 2.5,
        'training_complexity': 'high'
    }
}

# ============================================================================
# OPTIMIZER CONFIGURATIONS
# ============================================================================

# Supported optimizers dengan configurations
OPTIMIZER_CONFIGS = {
    'SGD': {
        'class_name': 'torch.optim.SGD',
        'default_params': {
            'lr': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'nesterov': False
        },
        'tunable_params': ['lr', 'momentum', 'weight_decay'],
        'description': 'Stochastic Gradient Descent (recommended untuk YOLOv5)',
        'memory_efficient': True
    },
    'Adam': {
        'class_name': 'torch.optim.Adam',
        'default_params': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'weight_decay': 0.0005,
            'eps': 1e-8
        },
        'tunable_params': ['lr', 'weight_decay'],
        'description': 'Adaptive Moment Estimation (good untuk EfficientNet)',
        'memory_efficient': False
    },
    'AdamW': {
        'class_name': 'torch.optim.AdamW',
        'default_params': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'weight_decay': 0.01,
            'eps': 1e-8
        },
        'tunable_params': ['lr', 'weight_decay'],
        'description': 'Adam dengan weight decay yang diperbaiki',
        'memory_efficient': False
    }
}

# Scheduler configurations
SCHEDULER_CONFIGS = {
    'cosine': {
        'class_name': 'torch.optim.lr_scheduler.CosineAnnealingLR',
        'description': 'Cosine annealing learning rate (recommended)',
        'params': {'T_max': 'epochs', 'eta_min': 0}
    },
    'step': {
        'class_name': 'torch.optim.lr_scheduler.StepLR',
        'description': 'Step learning rate decay',
        'params': {'step_size': 30, 'gamma': 0.1}
    },
    'exponential': {
        'class_name': 'torch.optim.lr_scheduler.ExponentialLR',
        'description': 'Exponential learning rate decay',
        'params': {'gamma': 0.9}
    }
}

# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

# Default configuration untuk training UI
DEFAULT_TRAINING_UI_CONFIG = {
    ConfigCategory.MODEL: {
        'model_type': 'efficient_optimized',
        'backbone': 'efficientnet_b4',
        'detection_layers': ['banknote'],
        'layer_mode': 'single',
        'num_classes': 7,
        'image_size': 640,
        'transfer_learning': True
    },
    ConfigCategory.TRAINING: {
        'epochs': 100,
        'batch_size': 16,
        'early_stopping': True,
        'patience': 10,
        'save_best': True,
        'save_interval': 10,
        'validation_split': 0.2
    },
    ConfigCategory.HYPERPARAMETERS: {
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'optimizer': 'Adam',
        'scheduler': 'cosine',
        'momentum': 0.937,
        'warmup_epochs': 3
    },
    ConfigCategory.BACKBONE: {
        'type': 'efficientnet_b4',
        'pretrained': True,
        'freeze': False,
        'use_attention': True
    },
    ConfigCategory.PATHS: {
        'data_dir': '/data/preprocessed',
        'checkpoint_dir': 'runs/train/checkpoints',
        'tensorboard_dir': 'runs/tensorboard',
        'output_dir': 'output'
    },
    ConfigCategory.TRAINING_STRATEGY: {
        'multi_scale': False,
        'mosaic': True,
        'mixup': False,
        'copy_paste': False
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_type_config(model_type: str) -> ModelTypeConfig:
    """Get model type configuration"""
    return MODEL_TYPE_CONFIGURATIONS.get(model_type, MODEL_TYPE_CONFIGURATIONS['efficient_optimized'])

def validate_config_category(category: ConfigCategory, config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate configuration untuk category tertentu"""
    errors = []
    required_keys = REQUIRED_CONFIG_KEYS.get(category, [])
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    
    # Validate parameter constraints
    if category == ConfigCategory.TRAINING or category == ConfigCategory.HYPERPARAMETERS:
        for param, value in config.items():
            if param in TRAINING_PARAMETER_CONSTRAINTS:
                constraint = TRAINING_PARAMETER_CONSTRAINTS[param]
                if not isinstance(value, constraint.data_type):
                    errors.append(f"{param} must be {constraint.data_type.__name__}")
                elif not (constraint.min_value <= value <= constraint.max_value):
                    errors.append(f"{param} must be between {constraint.min_value} and {constraint.max_value}")
    
    return len(errors) == 0, errors

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configurations dengan priority order"""
    merged = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged

def get_recommended_config(model_type: str, layer_mode: str) -> Dict[str, Any]:
    """Get recommended configuration untuk model type dan layer mode"""
    base_config = DEFAULT_TRAINING_UI_CONFIG.copy()
    model_config = get_model_type_config(model_type)
    
    # Update berdasarkan model type
    base_config[ConfigCategory.MODEL]['model_type'] = model_type
    base_config[ConfigCategory.MODEL]['backbone'] = model_config.backbone
    base_config[ConfigCategory.TRAINING]['batch_size'] = model_config.recommended_batch_size
    base_config[ConfigCategory.HYPERPARAMETERS]['learning_rate'] = model_config.recommended_lr
    
    # Update berdasarkan layer mode
    base_config[ConfigCategory.MODEL]['layer_mode'] = layer_mode
    if layer_mode == 'multilayer':
        base_config[ConfigCategory.MODEL]['detection_layers'] = ['banknote', 'nominal']
        base_config[ConfigCategory.TRAINING]['batch_size'] = max(8, model_config.recommended_batch_size // 2)
    
    return base_config

def get_total_classes(detection_layers: List[str]) -> int:
    """Calculate total classes dari detection layers"""
    total = 0
    for layer in detection_layers:
        if layer in DETECTION_LAYER_CONFIGS:
            total += DETECTION_LAYER_CONFIGS[layer]['num_classes']
    return total

# Export key constants
__all__ = [
    'ConfigSource', 'ConfigCategory', 'YAML_CONFIG_MAPPING', 'MODEL_TYPE_CONFIGURATIONS',
    'BACKBONE_CONFIGURATIONS', 'DETECTION_LAYER_CONFIGS', 'LAYER_MODE_CONFIGS',
    'OPTIMIZER_CONFIGS', 'SCHEDULER_CONFIGS', 'DEFAULT_TRAINING_UI_CONFIG',
    'TRAINING_PARAMETER_CONSTRAINTS', 'get_model_type_config', 'validate_config_category',
    'merge_configs', 'get_recommended_config', 'get_total_classes'
]