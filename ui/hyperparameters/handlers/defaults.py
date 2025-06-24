# File: smartcash/ui/hyperparameters/handlers/defaults.py
# Deskripsi: Default hyperparameters config yang disederhanakan sesuai backend usage

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Ambil konfigurasi default hyperparameters untuk SmartCash model 🎯"""
    return {
        # Training configuration - backend essentials
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.01,
            'image_size': 640,
            'device': 'auto',
            'workers': 4
        },
        
        # Optimizer configuration - backend essentials
        'optimizer': {
            'type': 'SGD',
            'weight_decay': 0.0005,
            'momentum': 0.937
        },
        
        # Learning rate scheduler - backend essentials
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 3,
            'min_lr': 0.0001
        },
        
        # Loss function weights - backend essentials
        'loss': {
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0
        },
        
        # Early stopping - backend essentials
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 0.001,
            'metric': 'mAP_0.5'
        },
        
        # Checkpoint configuration - backend essentials
        'checkpoint': {
            'save_best': True,
            'save_interval': 10,
            'metric': 'mAP_0.5'
        },
        
        # Model inference parameters - backend essentials
        'model': {
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'max_det': 1000
        },
        
        # Config metadata
        'config_version': '2.2',
        'description': 'Simplified hyperparameters untuk SmartCash YOLOv5-EfficientNet',
        'module_name': 'hyperparameters'
    }


def get_hyperparameters_ui_config() -> Dict[str, Any]:
    """Konfigurasi UI elements untuk hyperparameters form 🖥️"""
    return {
        'form_sections': [
            {
                'title': '🏋️ Training Parameters',
                'fields': ['epochs', 'batch_size', 'learning_rate', 'image_size', 'workers']
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
                'title': '⏹️ Early Stopping & Checkpoint',
                'fields': ['early_stopping_enabled', 'patience', 'save_best', 'save_interval']
            },
            {
                'title': '🔍 Model Inference',
                'fields': ['conf_thres', 'iou_thres', 'max_det']
            }
        ],
        'field_configs': {
            # Training parameters
            'epochs': {'type': 'IntSlider', 'min': 50, 'max': 300, 'step': 10},
            'batch_size': {'type': 'Dropdown', 'options': [8, 16, 32, 64]},
            'learning_rate': {'type': 'FloatLogSlider', 'min': 1e-4, 'max': 1e-1, 'step': 0.001},
            'image_size': {'type': 'Dropdown', 'options': [416, 512, 608, 640]},
            'workers': {'type': 'IntSlider', 'min': 0, 'max': 8, 'step': 1},
            
            # Optimizer parameters
            'optimizer_type': {'type': 'Dropdown', 'options': ['SGD', 'Adam']},
            'weight_decay': {'type': 'FloatLogSlider', 'min': 1e-6, 'max': 1e-2, 'step': 0.0001},
            'momentum': {'type': 'FloatSlider', 'min': 0.8, 'max': 0.99, 'step': 0.01},
            
            # Scheduler parameters
            'scheduler_type': {'type': 'Dropdown', 'options': ['cosine', 'step', 'plateau']},
            'warmup_epochs': {'type': 'IntSlider', 'min': 0, 'max': 10, 'step': 1},
            'min_lr': {'type': 'FloatLogSlider', 'min': 1e-6, 'max': 1e-3, 'step': 0.00001},
            
            # Loss parameters
            'box_loss_gain': {'type': 'FloatSlider', 'min': 0.01, 'max': 0.2, 'step': 0.01},
            'cls_loss_gain': {'type': 'FloatSlider', 'min': 0.1, 'max': 1.0, 'step': 0.1},
            'obj_loss_gain': {'type': 'FloatSlider', 'min': 0.5, 'max': 2.0, 'step': 0.1},
            
            # Early stopping parameters
            'early_stopping_enabled': {'type': 'Checkbox'},
            'patience': {'type': 'IntSlider', 'min': 5, 'max': 50, 'step': 5},
            'save_best': {'type': 'Checkbox'},
            'save_interval': {'type': 'IntSlider', 'min': 5, 'max': 50, 'step': 5},
            
            # Model inference parameters
            'conf_thres': {'type': 'FloatSlider', 'min': 0.1, 'max': 0.9, 'step': 0.05},
            'iou_thres': {'type': 'FloatSlider', 'min': 0.1, 'max': 0.9, 'step': 0.05},
            'max_det': {'type': 'IntSlider', 'min': 100, 'max': 2000, 'step': 100}
        }
    }