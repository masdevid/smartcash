# File: smartcash/ui/hyperparameters/handlers/defaults.py
# Deskripsi: Konfigurasi default hyperparameters yang aman dan teruji

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Ambil konfigurasi default hyperparameters untuk SmartCash model 🎯"""
    return {
        # Parameter training utama
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.01,
            'image_size': 640,
            'device': 'auto',  # auto-detect CUDA/CPU
            'workers': 4,
            'patience': 15
        },
        
        # Optimizer configuration
        'optimizer': {
            'type': 'SGD',
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'nesterov': True
        },
        
        # Learning rate scheduler
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 3,
            'min_lr': 0.0001,
            'T_max': 100
        },
        
        # Loss function weights
        'loss': {
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0,
            'label_smoothing': 0.0
        },
        
        # Early stopping
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 0.001,
            'metric': 'mAP_0.5'
        },
        
        # Checkpoint configuration
        'checkpoint': {
            'save_best': True,
            'save_interval': 10,
            'metric': 'mAP_0.5',
            'mode': 'max'
        },
        
        # Model configuration
        'model': {
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'max_det': 1000,
            'agnostic_nms': False
        },
        
        # Data augmentation (inline dengan training)
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0
        },
        
        # Config metadata
        'config_version': '2.1',
        'description': 'Default hyperparameters untuk SmartCash YOLOv5-EfficientNet',
        'module_name': 'hyperparameters'
    }


def get_hyperparameters_ui_config() -> Dict[str, Any]:
    """Konfigurasi UI elements untuk hyperparameters form 🖥️"""
    return {
        'form_sections': [
            {
                'title': '🏋️ Training Parameters',
                'fields': ['epochs', 'batch_size', 'learning_rate', 'image_size']
            },
            {
                'title': '⚙️ Optimizer Settings', 
                'fields': ['optimizer_type', 'weight_decay', 'momentum']
            },
            {
                'title': '📈 Learning Rate Scheduler',
                'fields': ['scheduler_type', 'warmup_epochs', 'min_lr']
            },
            {
                'title': '🎯 Loss Configuration',
                'fields': ['box_loss_gain', 'cls_loss_gain', 'obj_loss_gain']
            },
            {
                'title': '⏹️ Early Stopping',
                'fields': ['early_stopping_enabled', 'patience', 'min_delta']
            }
        ],
        'field_configs': {
            'epochs': {'type': 'IntSlider', 'min': 10, 'max': 300, 'step': 10},
            'batch_size': {'type': 'Dropdown', 'options': [8, 16, 32, 64]},
            'learning_rate': {'type': 'FloatLogSlider', 'min': 1e-5, 'max': 1e-1, 'step': 0.001},
            'image_size': {'type': 'Dropdown', 'options': [416, 512, 608, 640]},
            'optimizer_type': {'type': 'Dropdown', 'options': ['SGD', 'Adam', 'AdamW']},
            'weight_decay': {'type': 'FloatLogSlider', 'min': 1e-6, 'max': 1e-2, 'step': 0.0001},
            'momentum': {'type': 'FloatSlider', 'min': 0.0, 'max': 1.0, 'step': 0.1},
            'scheduler_type': {'type': 'Dropdown', 'options': ['cosine', 'step', 'plateau']},
            'warmup_epochs': {'type': 'IntSlider', 'min': 0, 'max': 10, 'step': 1},
            'min_lr': {'type': 'FloatLogSlider', 'min': 1e-6, 'max': 1e-3, 'step': 0.00001},
            'patience': {'type': 'IntSlider', 'min': 5, 'max': 50, 'step': 5}
        }
    }


def validate_hyperparameters_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """Validasi konfigurasi hyperparameters ✅"""
    try:
        # Validasi struktur utama
        required_sections = ['training', 'optimizer', 'scheduler', 'loss', 'early_stopping']
        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"
        
        # Validasi training parameters
        training = config['training']
        if training['epochs'] <= 0 or training['batch_size'] <= 0:
            return False, "Training epochs and batch_size must be positive"
        
        if not (0 < training['learning_rate'] <= 1):
            return False, "Learning rate must be between 0 and 1"
        
        # Validasi optimizer
        optimizer = config['optimizer']
        if optimizer['type'] not in ['SGD', 'Adam', 'AdamW']:
            return False, "Invalid optimizer type"
        
        # Validasi scheduler
        scheduler = config['scheduler']
        if scheduler['type'] not in ['cosine', 'step', 'plateau']:
            return False, "Invalid scheduler type"
        
        return True, "Configuration is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
