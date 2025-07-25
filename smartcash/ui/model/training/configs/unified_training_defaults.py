"""
File: smartcash/ui/model/training/configs/unified_training_defaults.py
Description: Default configuration for unified training pipeline UI.
"""

from typing import Dict, Any


def get_unified_training_defaults() -> Dict[str, Any]:
    """Get default configuration matching unified_training_example.py parameters.
    
    Returns:
        Default training configuration dictionary
    """
    return {
        'training': {
            # Core parameters matching unified_training_example.py
            'backbone': 'cspdarknet',  # Limited to efficientnet_b4 and cspdarknet
            'training_mode': 'two_phase',  # two_phase or single_phase
            'phase_1_epochs': 1,
            'phase_2_epochs': 1,
            'checkpoint_dir': 'data/checkpoints',
            
            # Training configuration parameters
            'loss_type': 'uncertainty_multi_task',
            'head_lr_p1': 1e-3,
            'head_lr_p2': 1e-4,
            'backbone_lr': 1e-5,
            'batch_size': None,  # Auto-detection based on platform
            
            # Early stopping configuration
            'early_stopping_enabled': True,
            'early_stopping_patience': 15,
            'early_stopping_metric': 'val_map50',
            'early_stopping_mode': 'max',
            'early_stopping_min_delta': 0.001,
            
            # Single-phase specific options
            'single_phase_layer_mode': 'multi',  # single or multi
            'single_phase_freeze_backbone': False,
            
            # System options
            'force_cpu': False,
            'verbose': True
        },
        
        # UI specific configuration
        'ui': {
            'show_advanced_options': True,
            'auto_open_basic_section': True,
            'enable_progress_tracking': True,
            'max_log_entries': 200
        }
    }


def get_backbone_options() -> Dict[str, str]:
    """Get available backbone options (limited as requested).
    
    Returns:
        Dictionary mapping backbone keys to display names
    """
    return {
        'cspdarknet': 'YOLOv5s (CSPDarkNet)',
        'efficientnet_b4': 'EfficientNet-B4'
    }


def get_training_mode_options() -> Dict[str, str]:
    """Get available training mode options.
    
    Returns:
        Dictionary mapping training mode keys to display names
    """
    return {
        'two_phase': 'Two-Phase (Freeze → Fine-tune)',
        'single_phase': 'Single-Phase (Unified)'
    }


def get_loss_type_options() -> Dict[str, str]:
    """Get available loss type options.
    
    Returns:
        Dictionary mapping loss type keys to display names
    """
    return {
        'uncertainty_multi_task': 'Uncertainty Multi-Task',
        'weighted_multi_task': 'Weighted Multi-Task',
        'focal': 'Focal',
        'standard': 'Standard'
    }


def get_early_stopping_metric_options() -> Dict[str, str]:
    """Get available early stopping metric options.
    
    Returns:
        Dictionary mapping metric keys to display names
    """
    return {
        'val_map50': 'Validation mAP@0.5',
        'val_loss': 'Validation Loss',
        'train_loss': 'Training Loss',
        'val_map75': 'Validation mAP@0.75',
        'val_accuracy': 'Validation Accuracy'
    }


def get_early_stopping_mode_options() -> Dict[str, str]:
    """Get available early stopping mode options.
    
    Returns:
        Dictionary mapping mode keys to display names
    """
    return {
        'max': 'Maximize (for mAP, accuracy)',
        'min': 'Minimize (for loss)'
    }


def get_single_phase_layer_mode_options() -> Dict[str, str]:
    """Get available single-phase layer mode options.
    
    Returns:
        Dictionary mapping layer mode keys to display names
    """
    return {
        'multi': 'Multi-layer (all layers)',
        'single': 'Single-layer (layer_1 only)'
    }


def validate_unified_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate unified training configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation result with success status and messages
    """
    errors = []
    warnings = []
    
    training_config = config.get('training', {})
    
    # Validate backbone
    backbone = training_config.get('backbone')
    if backbone not in get_backbone_options():
        errors.append(f"Invalid backbone: {backbone}. Must be one of: {list(get_backbone_options().keys())}")
    
    # Validate training mode
    training_mode = training_config.get('training_mode')
    if training_mode not in get_training_mode_options():
        errors.append(f"Invalid training mode: {training_mode}. Must be one of: {list(get_training_mode_options().keys())}")
    
    # Validate epochs
    phase_1_epochs = training_config.get('phase_1_epochs', 1)
    phase_2_epochs = training_config.get('phase_2_epochs', 1)
    
    if not isinstance(phase_1_epochs, int) or phase_1_epochs < 1:
        errors.append("Phase 1 epochs must be a positive integer")
    elif phase_1_epochs > 20:
        warnings.append(f"Phase 1 epochs ({phase_1_epochs}) is quite high. Consider lower values for faster training.")
    
    if not isinstance(phase_2_epochs, int) or phase_2_epochs < 1:
        errors.append("Phase 2 epochs must be a positive integer")
    elif phase_2_epochs > 20:
        warnings.append(f"Phase 2 epochs ({phase_2_epochs}) is quite high. Consider lower values for faster training.")
    
    # Validate learning rates
    head_lr_p1 = training_config.get('head_lr_p1', 1e-3)
    head_lr_p2 = training_config.get('head_lr_p2', 1e-4)
    backbone_lr = training_config.get('backbone_lr', 1e-5)
    
    if not isinstance(head_lr_p1, (int, float)) or head_lr_p1 <= 0:
        errors.append("Head LR (Phase 1) must be a positive number")
    elif head_lr_p1 > 0.1:
        warnings.append(f"Head LR Phase 1 ({head_lr_p1}) is quite high. Consider values around 1e-3.")
    
    if not isinstance(head_lr_p2, (int, float)) or head_lr_p2 <= 0:
        errors.append("Head LR (Phase 2) must be a positive number")
    elif head_lr_p2 > head_lr_p1:
        warnings.append("Head LR Phase 2 should typically be lower than Phase 1 LR")
    
    if not isinstance(backbone_lr, (int, float)) or backbone_lr <= 0:
        errors.append("Backbone LR must be a positive number")
    elif backbone_lr > head_lr_p2:
        warnings.append("Backbone LR should typically be lower than head learning rates")
    
    # Validate batch size
    batch_size = training_config.get('batch_size')
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("Batch size must be a positive integer or None for auto-detection")
        elif batch_size > 32:
            warnings.append(f"Batch size ({batch_size}) is quite large. Consider lower values if memory issues occur.")
    
    # Validate early stopping
    if training_config.get('early_stopping_enabled', True):
        patience = training_config.get('early_stopping_patience', 15)
        if not isinstance(patience, int) or patience < 1:
            errors.append("Early stopping patience must be a positive integer")
        
        metric = training_config.get('early_stopping_metric')
        if metric not in get_early_stopping_metric_options():
            errors.append(f"Invalid early stopping metric: {metric}")
        
        mode = training_config.get('early_stopping_mode')
        if mode not in get_early_stopping_mode_options():
            errors.append(f"Invalid early stopping mode: {mode}")
    
    return {
        'success': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'message': f"Validation {'passed' if len(errors) == 0 else 'failed'} with {len(errors)} errors and {len(warnings)} warnings"
    }