"""
File: smartcash/ui/training/constants/training_ui_constants.py
Deskripsi: Enhanced constants untuk training UI dengan YAML config alignment
"""

from typing import Dict, Any, List, Tuple

# ============================================================================
# TRAINING UI MODULE INFO
# ============================================================================

TRAINING_UI_MODULE_NAME = "SmartCash Training UI"
TRAINING_UI_VERSION = "2.0.0"
TRAINING_UI_NAMESPACE = "smartcash.ui.training"

# ============================================================================
# YAML CONFIG ALIGNMENT CONSTANTS
# ============================================================================

# Model types sesuai model_config.yaml
SUPPORTED_MODEL_TYPES = {
    'efficient_basic': {
        'name': 'EfficientNet-B4 Basic',
        'backbone': 'efficientnet_b4',
        'description': 'Model dasar dengan EfficientNet-B4 tanpa optimasi khusus',
        'optimizations': {'use_attention': False, 'use_residual': False, 'use_ciou': False}
    },
    'efficient_optimized': {
        'name': 'EfficientNet-B4 Optimized', 
        'backbone': 'efficientnet_b4',
        'description': 'Model dengan optimasi feature maps',
        'optimizations': {'use_attention': True, 'use_residual': False, 'use_ciou': False}
    },
    'efficient_advanced': {
        'name': 'EfficientNet-B4 Advanced',
        'backbone': 'efficientnet_b4', 
        'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
        'optimizations': {'use_attention': True, 'use_residual': True, 'use_ciou': True}
    },
    'yolov5s': {
        'name': 'YOLOv5s Baseline',
        'backbone': 'cspdarknet_s',
        'description': 'YOLOv5s dengan CSPDarknet sebagai backbone (model pembanding)',
        'optimizations': {'use_attention': False, 'use_residual': False, 'use_ciou': False}
    }
}

# Backbone configurations sesuai YAML
SUPPORTED_BACKBONES = {
    'efficientnet_b4': {
        'name': 'EfficientNet-B4',
        'input_size': [640, 640],
        'pretrained': True,
        'width_coefficient': 1.4,
        'depth_coefficient': 1.8,
        'dropout_rate': 0.4,
        'recommended_batch_size': 16,
        'recommended_lr': 0.001
    },
    'cspdarknet_s': {
        'name': 'CSPDarknet-S',
        'input_size': [640, 640], 
        'pretrained': True,
        'depth_multiple': 0.67,
        'width_multiple': 0.75,
        'recommended_batch_size': 32,
        'recommended_lr': 0.01
    }
}

# Training parameters sesuai hyperparameters_config.yaml
DEFAULT_TRAINING_PARAMS = {
    'batch_size': 16,
    'image_size': 640,
    'epochs': 100,
    'optimizer': 'Adam',
    'learning_rate': 0.01,
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'scheduler': 'cosine',
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'augment': True,
    'dropout': 0.0,
    'box_loss_gain': 0.05,
    'cls_loss_gain': 0.5,
    'obj_loss_gain': 1.0,
    'anchor_t': 4.0,
    'fl_gamma': 0.0
}

# Early stopping & save best sesuai YAML
TRAINING_STRATEGY_PARAMS = {
    'early_stopping': {
        'enabled': True,
        'patience': 15,
        'min_delta': 0.001
    },
    'save_best': {
        'enabled': True,
        'metric': 'mAP_0.5'
    },
    'validation': {
        'frequency': 1,
        'iou_thres': 0.6,
        'conf_thres': 0.001
    },
    'multi_scale': True
}

# Training utils sesuai training_config.yaml
TRAINING_UTILS_PARAMS = {
    'experiment_name': 'efficientnet_b4_training',
    'checkpoint_dir': '/content/runs/train/checkpoints',
    'tensorboard': True,
    'log_metrics_every': 10,
    'visualize_batch_every': 100,
    'gradient_clipping': 1.0,
    'mixed_precision': True,
    'layer_mode': 'single'
}

# ============================================================================
# UI COMPONENT CONFIGURATIONS
# ============================================================================

# Critical UI components untuk validation
CRITICAL_UI_COMPONENTS = [
    'main_container',
    'start_button', 
    'stop_button',
    'config_tabs',
    'log_output'
]

# Optional UI components
OPTIONAL_UI_COMPONENTS = [
    'reset_button',
    'validate_button',
    'refresh_button',
    'progress_container',
    'chart_output',
    'metrics_output',
    'status_panel'
]

# Button configurations sesuai training state
BUTTON_CONFIGS = {
    'start_training': {
        'idle': {'description': '🚀 Mulai Training', 'disabled': False, 'style': 'success'},
        'running': {'description': '🔄 Training...', 'disabled': True, 'style': 'warning'},
        'error': {'description': '🔄 Retry Training', 'disabled': False, 'style': 'danger'}
    },
    'stop_training': {
        'idle': {'description': '⏹️ Stop Training', 'disabled': True, 'style': ''},
        'running': {'description': '⏹️ Stop Training', 'disabled': False, 'style': 'danger'}
    },
    'reset_metrics': {
        'idle': {'description': '🔄 Reset Metrics', 'disabled': False, 'style': 'warning'},
        'running': {'description': '🔄 Reset Metrics', 'disabled': True, 'style': ''}
    }
}

# ============================================================================
# CHART & METRICS CONFIGURATIONS
# ============================================================================

# Chart configuration untuk matplotlib
CHART_CONFIG = {
    'figure_size': (12, 4),
    'subplot_layout': (1, 2),
    'line_width': 2,
    'grid_alpha': 0.3,
    'title_size': 12,
    'label_size': 10,
    'dpi': 100
}

# Chart colors untuk different metrics
CHART_COLORS = {
    'train_loss': '#007bff',
    'val_loss': '#dc3545', 
    'learning_rate': '#28a745',
    'mAP': '#17a2b8',
    'precision': '#6f42c1',
    'recall': '#e83e8c',
    'f1_score': '#fd7e14',
    'box_loss': '#20c997',
    'obj_loss': '#ffc107',
    'cls_loss': '#e83e8c'
}

# Metrics display configuration
METRICS_CONFIG = {
    'precision': 4,
    'scientific_threshold': 0.01,
    'percentage_metrics': ['mAP', 'precision', 'recall', 'f1_score'],
    'loss_metrics': ['loss', 'train_loss', 'val_loss', 'box_loss', 'obj_loss', 'cls_loss'],
    'time_metrics': ['epoch_time', 'batch_time', 'inference_time'],
    'lr_metrics': ['learning_rate', 'lr']
}

# ============================================================================
# VALIDATION RULES SESUAI YAML
# ============================================================================

# Parameter validation rules
VALIDATION_RULES = {
    'epochs': {'min': 1, 'max': 1000, 'type': int, 'yaml_default': 100},
    'batch_size': {'min': 1, 'max': 128, 'type': int, 'yaml_default': 16},
    'learning_rate': {'min': 1e-6, 'max': 1.0, 'type': float, 'yaml_default': 0.01},
    'weight_decay': {'min': 0.0, 'max': 1.0, 'type': float, 'yaml_default': 0.0005},
    'momentum': {'min': 0.0, 'max': 1.0, 'type': float, 'yaml_default': 0.937},
    'warmup_epochs': {'min': 0, 'max': 20, 'type': int, 'yaml_default': 3},
    'patience': {'min': 1, 'max': 100, 'type': int, 'yaml_default': 15},
    'image_size': {'min': 320, 'max': 1280, 'type': int, 'yaml_default': 640}
}

# Required config keys per category
REQUIRED_CONFIG_KEYS = {
    'model': ['type', 'backbone', 'backbone_pretrained'],
    'training': ['epochs', 'batch_size'],
    'hyperparameters': ['learning_rate', 'weight_decay', 'optimizer'],
    'training_utils': ['experiment_name', 'checkpoint_dir', 'layer_mode']
}

# ============================================================================
# LOG MESSAGE TEMPLATES
# ============================================================================

# Log templates dengan YAML context
LOG_TEMPLATES = {
    'TRAINING_START': "🚀 Training {model_type} dimulai: {epochs} epochs, batch {batch_size}, {optimizer} optimizer",
    'EPOCH_COMPLETE': "✅ Epoch {epoch}/{total_epochs} selesai - Loss: {loss:.4f}, mAP: {map:.4f}",
    'BEST_MODEL': "🏆 New best model: {metric}={value:.4f} (improved by {improvement:.2f}%)",
    'TRAINING_COMPLETE': "🎉 Training selesai! Best epoch: {best_epoch}, Experiment: {experiment_name}",
    'CONFIG_REFRESH': "🔄 YAML Configuration refreshed: {config_files}",
    'MODEL_VALIDATION': "🔍 Model validation: {model_type} dengan {backbone}",
    'EARLY_STOPPING': "⏸️ Early stopping triggered at epoch {epoch} (patience: {patience})",
    'CHECKPOINT_SAVED': "💾 Checkpoint saved: {checkpoint_path} (epoch {epoch})"
}

# Progress message templates
PROGRESS_TEMPLATES = {
    'INITIALIZING': "🔄 Initializing {component} dengan YAML config...",
    'TRAINING_EPOCH': "📅 Training Epoch {epoch}/{total_epochs} ({model_type})",
    'VALIDATION': "🔍 Validating epoch {epoch} - {validation_metric}",
    'SAVING_CHECKPOINT': "💾 Saving checkpoint epoch {epoch} ke {checkpoint_dir}",
    'LOADING_MODEL': "📂 Loading {model_type} dengan {backbone} backbone",
    'YAML_CONFIG_LOADED': "📄 YAML configs loaded: {config_count} parameters"
}

# ============================================================================
# ERROR CATEGORIES & RECOVERY
# ============================================================================

# Error categories sesuai training components
ERROR_CATEGORIES = {
    'YAML_CONFIG_ERROR': 'yaml_config_error',
    'MODEL_INIT_ERROR': 'model_init_error', 
    'TRAINING_ERROR': 'training_error',
    'UI_COMPONENT_ERROR': 'ui_component_error'
}

# Error recovery actions
ERROR_RECOVERY = {
    'YAML_CONFIG_ERROR': 'Check YAML files syntax dan parameter values',
    'MODEL_INIT_ERROR': 'Verify model type dan backbone compatibility',
    'TRAINING_ERROR': 'Check training parameters dan data loader',
    'UI_COMPONENT_ERROR': 'Restart training UI atau check component dependencies'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_type_info(model_type: str) -> Dict[str, Any]:
    """Get model type configuration info"""
    return SUPPORTED_MODEL_TYPES.get(model_type, SUPPORTED_MODEL_TYPES['efficient_basic'])

def get_backbone_info(backbone: str) -> Dict[str, Any]:
    """Get backbone configuration info"""
    return SUPPORTED_BACKBONES.get(backbone, SUPPORTED_BACKBONES['efficientnet_b4'])

def validate_yaml_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate parameters against YAML rules"""
    errors = []
    
    for param, value in params.items():
        if param in VALIDATION_RULES:
            rule = VALIDATION_RULES[param]
            
            if not isinstance(value, rule['type']):
                errors.append(f"{param} harus bertipe {rule['type'].__name__}")
                continue
                
            if value < rule['min'] or value > rule['max']:
                errors.append(f"{param} harus antara {rule['min']} - {rule['max']}")
    
    return len(errors) == 0, errors

def get_yaml_default_config() -> Dict[str, Any]:
    """Get default config sesuai YAML structure"""
    return {
        **DEFAULT_TRAINING_PARAMS,
        **TRAINING_STRATEGY_PARAMS,
        **TRAINING_UTILS_PARAMS,
        'model': {
            'type': 'efficient_basic',
            'backbone': 'efficientnet_b4',
            'backbone_pretrained': True,
            'confidence': 0.25,
            'iou_threshold': 0.45
        }
    }

def get_button_config(button_name: str, state: str) -> Dict[str, Any]:
    """Get button configuration for specific state"""
    return BUTTON_CONFIGS.get(button_name, {}).get(state, {
        'description': button_name, 'disabled': False, 'style': ''
    })

def format_training_log(template_key: str, **kwargs) -> str:
    """Format training log message dengan template"""
    template = LOG_TEMPLATES.get(template_key, "{message}")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template

def get_chart_color(metric_name: str) -> str:
    """Get chart color untuk specific metric"""
    return CHART_COLORS.get(metric_name, '#6c757d')

def is_percentage_metric(metric_name: str) -> bool:
    """Check apakah metric adalah percentage"""
    return metric_name in METRICS_CONFIG['percentage_metrics']

def is_loss_metric(metric_name: str) -> bool:
    """Check apakah metric adalah loss"""
    return any(loss in metric_name.lower() for loss in METRICS_CONFIG['loss_metrics'])

# Export key constants
__all__ = [
    'SUPPORTED_MODEL_TYPES', 'SUPPORTED_BACKBONES', 'DEFAULT_TRAINING_PARAMS',
    'TRAINING_STRATEGY_PARAMS', 'TRAINING_UTILS_PARAMS', 'CRITICAL_UI_COMPONENTS',
    'BUTTON_CONFIGS', 'CHART_CONFIG', 'METRICS_CONFIG', 'VALIDATION_RULES',
    'LOG_TEMPLATES', 'ERROR_CATEGORIES', 'get_model_type_info', 'get_backbone_info',
    'validate_yaml_params', 'get_yaml_default_config', 'get_button_config',
    'format_training_log', 'get_chart_color'
]