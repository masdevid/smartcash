"""
File: smartcash/ui/model/train/configs/train_defaults.py
Default configuration for train module following UI module structure standard.
"""

from typing import Dict, Any
from ..constants import DEFAULT_CONFIG, LAYER_CONFIGS, LayerMode

def get_default_train_config() -> Dict[str, Any]:
    """
    Get default training configuration.
    
    Returns:
        Dict containing default training configuration with all required sections
    """
    return {
        'training': {
            'layer_mode': 'single',
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
            'gradient_accumulation': 1,
            'backbone_integration': {
                'load_from_backbone': True,
                'inherit_config': True,
                'validate_compatibility': True
            }
        },
        'optimizer': {
            'type': 'adam',
            'weight_decay': 0.0005,
            'momentum': 0.9,
            'adaptive_lr': True
        },
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 5,
            'min_lr': 0.00001,
            'cycle_momentum': True
        },
        'model_storage': {
            'checkpoints_dir': '/data/checkpoints',
            'best_models_dir': '/data/models/best',
            'logs_dir': '/data/logs/training',
            'tensorboard_dir': '/data/logs/tensorboard',
            'auto_save_best': True,
            'overwrite_existing': True  # No history tracking due to storage constraints
        },
        'monitoring': {
            'live_charts_enabled': True,
            'progress_updates_enabled': True,
            'chart_update_interval': 1000,
            'metrics_logging': True,
            'tensorboard_logging': False
        },
        'ui': {
            'show_advanced_options': False,
            'show_layer_options': True,
            'dual_charts_layout': 'horizontal',
            'auto_refresh_charts': True
        }
    }

def get_layer_mode_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get layer mode configurations.
    
    Returns:
        Dict containing layer mode configurations
    """
    return {
        'single': {
            'display_name': 'Single Layer Training',
            'description': 'Train single detection layer (banknote only)',
            'layers': ['banknote'],
            'num_classes': 1,
            'recommended_epochs': 100,
            'recommended_lr': 0.001,
            'recommended_batch_size': 16,
            'complexity': 'Low',
            'training_time': 'Fast'
        },
        'multilayer': {
            'display_name': 'Multi-Layer Training',
            'description': 'Train multiple detection layers (banknote + nominal + security)',
            'layers': ['banknote', 'nominal', 'security'],
            'num_classes': 3,
            'recommended_epochs': 150,
            'recommended_lr': 0.0008,
            'recommended_batch_size': 12,
            'complexity': 'High',
            'training_time': 'Slower'
        }
    }

def get_optimization_types() -> Dict[str, Dict[str, Any]]:
    """
    Get optimization type configurations.
    
    Returns:
        Dict containing optimization type configurations
    """
    return {
        'default': {
            'display_name': 'Default Optimization',
            'description': 'Standard training configuration',
            'learning_rate_factor': 1.0,
            'batch_size_factor': 1.0,
            'additional_augmentation': False,
            'advanced_techniques': False
        },
        'optimized': {
            'display_name': 'Optimized Training',
            'description': 'Enhanced training with optimizations',
            'learning_rate_factor': 0.8,
            'batch_size_factor': 1.2,
            'additional_augmentation': True,
            'advanced_techniques': True
        },
        'advanced': {
            'display_name': 'Advanced Training',
            'description': 'Advanced training with experimental features',
            'learning_rate_factor': 0.6,
            'batch_size_factor': 1.5,
            'additional_augmentation': True,
            'advanced_techniques': True
        }
    }

def get_training_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get training preset configurations.
    
    Returns:
        Dict containing training presets
    """
    return {
        'quick_test': {
            'display_name': 'Quick Test',
            'description': 'Fast training for testing (10 epochs)',
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 0.01,
            'validation_interval': 2,
            'save_interval': 5
        },
        'standard': {
            'display_name': 'Standard Training',
            'description': 'Balanced training configuration',
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'validation_interval': 1,
            'save_interval': 5
        },
        'high_quality': {
            'display_name': 'High Quality',
            'description': 'Long training for best results',
            'epochs': 200,
            'batch_size': 12,
            'learning_rate': 0.0008,
            'validation_interval': 1,
            'save_interval': 3
        }
    }

def get_backbone_integration_config() -> Dict[str, Any]:
    """
    Get backbone integration configuration.
    
    Returns:
        Dict containing backbone integration settings
    """
    return {
        'load_from_backbone': True,
        'inherit_config': True,
        'validate_compatibility': True,
        'backbone_freeze_epochs': 0,  # Epochs to freeze backbone layers
        'backbone_lr_factor': 0.1,   # Learning rate factor for backbone layers
        'feature_extraction_only': False
    }

def get_chart_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Get dual chart configurations for live monitoring.
    
    Returns:
        Dict containing chart configurations
    """
    return {
        'loss_chart': {
            'title': 'Training Loss',
            'metrics': ['train_loss', 'val_loss'],
            'colors': ['#ff6b6b', '#ff9ff3'],
            'y_label': 'Loss Value',
            'y_scale': 'linear',
            'update_interval': 1000,
            'max_points': 1000,
            'grid': True,
            'legend': True
        },
        'map_chart': {
            'title': 'mAP Performance',
            'metrics': ['val_map50', 'val_map75'],
            'colors': ['#4ecdc4', '#45b7d1'],
            'y_label': 'mAP Score',
            'y_scale': 'linear',
            'update_interval': 1000,
            'max_points': 1000,
            'grid': True,
            'legend': True
        }
    }

def get_model_naming_examples() -> Dict[str, str]:
    """
    Get model naming convention examples.
    
    Returns:
        Dict containing naming examples
    """
    return {
        'efficientnet_b4_single_default': 'EfficientNet-B4 + Single Layer + Default optimization',
        'efficientnet_b4_multilayer_optimized': 'EfficientNet-B4 + Multi-Layer + Optimized training',
        'cspdarknet_single_default': 'CSPDarknet + Single Layer + Default optimization',
        'cspdarknet_multilayer_advanced': 'CSPDarknet + Multi-Layer + Advanced training'
    }