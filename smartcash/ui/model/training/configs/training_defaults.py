"""
File: smartcash/ui/model/training/configs/training_defaults.py
Default configuration for training module.
"""

from typing import Dict, Any
from enum import Enum


class TrainingPhase(Enum):
    """Training phase enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating" 
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


def get_default_training_config() -> Dict[str, Any]:
    """
    Get default training configuration.
    
    Returns:
        Default training configuration dictionary
    """
    return {
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 5e-4,  # AdamW default learning rate
            'optimizer': 'adamw',   # Default to AdamW
            'scheduler': 'cosine',  # CosineAnnealingLR as default
            'weight_decay': 1e-2,   # AdamW standard weight decay
            'warmup_epochs': 3,
            'save_period': 10,
            'val_period': 5,
            # AdamW specific parameters
            'adamw_betas': [0.9, 0.999],
            'adamw_eps': 1e-8,
            # CosineAnnealingLR specific parameters  
            'cosine_eta_min': 1e-6,
            'early_stopping': {
                'enabled': True,
                'patience': 15,
                'min_delta': 0.001
            },
            'mixed_precision': True,
            'gradient_clip': 1.0,
            'resume': False
        },
        'model_selection': {
            'auto_detect': True,
            'source': 'backbone',  # 'backbone', 'checkpoint', 'pretrained'
            'checkpoint_path': '',
            'validation_required': True
        },
        'data': {
            'train_split': 0.75,  # Fixed: 75% for training
            'val_split': 0.15,    # Fixed: 15% for validation  
            'test_split': 0.15,   # Fixed: 15% for testing (not configurable)
            'split_fixed': True,  # Indicates splits are fixed and not configurable
            'augmentation': {
                'enabled': True,
                'mixup': 0.1,
                'cutmix': 0.1,
                'mosaic': 0.5
            },
            'workers': 4
        },
        'monitoring': {
            'primary_metric': 'mAP@0.5',
            'watch_loss': True,
            'watch_map': True,
            'save_best_only': True,
            'tensorboard': False,
            'wandb': False
        },
        'charts': {
            'enabled': True,
            'update_frequency': 'epoch',  # 'batch', 'epoch'
            'max_points': 1000,
            'loss_chart': {
                'enabled': True,
                'title': 'Training & Validation Loss',
                'metrics': ['train_loss', 'val_loss']
            },
            'map_chart': {
                'enabled': True,
                'title': 'mAP Performance',
                'metrics': ['mAP@0.5', 'mAP@0.75']
            }
        },
        'output': {
            'save_dir': 'runs/training',
            'name': 'smartcash_training',
            'exist_ok': True,
            'save_txt': True,
            'save_conf': True
        },
        'device': {
            'auto_detect': True,
            'preferred': 'auto'  # 'auto', 'cpu', 'cuda'
        },
        'ui': {
            'show_advanced_options': False,
            'auto_start_validation': True,
            'show_progress_details': True,
            'enable_live_charts': True
        }
    }


def get_training_metrics_config() -> Dict[str, Any]:
    """Get training metrics configuration."""
    return {
        'loss_metrics': ['train_loss', 'val_loss'],
        'performance_metrics': ['mAP@0.5', 'mAP@0.75', 'precision', 'recall'],
        'learning_metrics': ['learning_rate', 'gradient_norm'],
        'system_metrics': ['gpu_memory', 'training_speed'],
        'default_values': {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'mAP@0.5': 0.0,
            'mAP@0.75': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'learning_rate': 5e-4,
            'gradient_norm': 0.0,
            'gpu_memory': 0.0,
            'training_speed': 0.0,
            'epoch': 0,
            'total_epochs': 100,
            'batch': 0,
            'total_batches': 0
        }
    }


def get_chart_config() -> Dict[str, Any]:
    """Get chart configuration for live updates."""
    return {
        'loss_chart': {
            'title': '📈 Training & Validation Loss',
            'x_label': 'Epoch',
            'y_label': 'Loss',
            'lines': {
                'train_loss': {'color': '#ff6b6b', 'label': 'Training Loss'},
                'val_loss': {'color': '#4ecdc4', 'label': 'Validation Loss'}
            },
            'height': '300px',
            'update_interval': 1000  # ms
        },
        'map_chart': {
            'title': '📊 mAP Performance',
            'x_label': 'Epoch', 
            'y_label': 'mAP',
            'lines': {
                'mAP@0.5': {'color': '#45b7d1', 'label': 'mAP@0.5'},
                'mAP@0.75': {'color': '#96ceb4', 'label': 'mAP@0.75'}
            },
            'height': '300px',
            'update_interval': 1000  # ms
        }
    }


def get_available_optimizers() -> Dict[str, str]:
    """Get available optimizer options."""
    return {
        'adam': 'Adam',
        'adamw': 'AdamW',
        'sgd': 'SGD with Momentum',
        'rmsprop': 'RMSprop'
    }


def get_available_schedulers() -> Dict[str, str]:
    """Get available learning rate scheduler options."""
    return {
        'cosine': 'Cosine Annealing',
        'step': 'Step Decay',
        'exp': 'Exponential Decay',
        'plateau': 'Reduce on Plateau',
        'none': 'No Scheduler'
    }


# Constants for training validation
TRAINING_VALIDATION_CONFIG = {
    'min_epochs': 1,
    'max_epochs': 1000,
    'min_batch_size': 1,
    'max_batch_size': 256,
    'min_learning_rate': 1e-6,
    'max_learning_rate': 1.0,
    'min_weight_decay': 0.0,
    'max_weight_decay': 1.0
}