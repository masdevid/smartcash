"""
File: smartcash/ui/model/backbone/utils/validation.py
Deskripsi: Validation utilities untuk backbone model configuration
"""

from typing import Dict, Any, List, Tuple

def validate_backbone_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Comprehensive validation untuk backbone configuration
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check root structure
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return False, errors
    
    if 'model' not in config:
        errors.append("Missing 'model' section in configuration")
        return False, errors
    
    model_config = config['model']
    
    # Validate backbone
    errors.extend(_validate_backbone(model_config))
    
    # Validate detection layers
    errors.extend(_validate_detection_layers(model_config))
    
    # Validate layer mode
    errors.extend(_validate_layer_mode(model_config))
    
    # Validate optimization settings
    errors.extend(_validate_optimization(model_config))
    
    # Validate numeric parameters
    errors.extend(_validate_numeric_params(model_config))
    
    return len(errors) == 0, errors

def _validate_backbone(config: Dict[str, Any]) -> List[str]:
    """Validate backbone selection"""
    errors = []
    
    if 'backbone' not in config:
        errors.append("Missing 'backbone' field")
        return errors
    
    valid_backbones = ['efficientnet_b4', 'cspdarknet']
    if config['backbone'] not in valid_backbones:
        errors.append(f"Invalid backbone: {config['backbone']}. Must be one of {valid_backbones}")
    
    return errors

def _validate_detection_layers(config: Dict[str, Any]) -> List[str]:
    """Validate detection layers configuration"""
    errors = []
    
    if 'detection_layers' not in config:
        errors.append("Missing 'detection_layers' field")
        return errors
    
    layers = config['detection_layers']
    if not isinstance(layers, list):
        errors.append("'detection_layers' must be a list")
        return errors
    
    if len(layers) == 0:
        errors.append("At least one detection layer must be selected")
    
    valid_layers = ['banknote', 'nominal', 'security']
    invalid_layers = [l for l in layers if l not in valid_layers]
    if invalid_layers:
        errors.append(f"Invalid detection layers: {invalid_layers}. Must be from {valid_layers}")
    
    return errors

def _validate_layer_mode(config: Dict[str, Any]) -> List[str]:
    """Validate layer mode configuration"""
    errors = []
    
    if 'layer_mode' not in config:
        errors.append("Missing 'layer_mode' field")
        return errors
    
    valid_modes = ['single', 'multilayer']
    if config['layer_mode'] not in valid_modes:
        errors.append(f"Invalid layer mode: {config['layer_mode']}. Must be one of {valid_modes}")
    
    # Check layer mode compatibility
    if config.get('layer_mode') == 'single' and len(config.get('detection_layers', [])) > 1:
        errors.append("Single layer mode selected but multiple detection layers specified")
    
    return errors

def _validate_optimization(config: Dict[str, Any]) -> List[str]:
    """Validate optimization settings"""
    errors = []
    
    # Feature optimization is optional
    if 'feature_optimization' in config:
        feat_opt = config['feature_optimization']
        if not isinstance(feat_opt, dict):
            errors.append("'feature_optimization' must be a dictionary")
        elif 'enabled' in feat_opt and not isinstance(feat_opt['enabled'], bool):
            errors.append("'feature_optimization.enabled' must be a boolean")
    
    # Mixed precision is optional
    if 'mixed_precision' in config and not isinstance(config['mixed_precision'], bool):
        errors.append("'mixed_precision' must be a boolean")
    
    return errors

def _validate_numeric_params(config: Dict[str, Any]) -> List[str]:
    """Validate numeric parameters"""
    errors = []
    
    # Validate num_classes
    if 'num_classes' in config:
        num_classes = config['num_classes']
        if not isinstance(num_classes, int) or num_classes < 1:
            errors.append("'num_classes' must be a positive integer")
        elif num_classes != 7:  # SmartCash specific
            errors.append("'num_classes' must be 7 for SmartCash currency detection")
    
    # Validate img_size
    if 'img_size' in config:
        img_size = config['img_size']
        if not isinstance(img_size, int) or img_size < 32:
            errors.append("'img_size' must be an integer >= 32")
        elif img_size % 32 != 0:
            errors.append("'img_size' must be divisible by 32")
        elif img_size not in [320, 416, 512, 640, 832]:
            errors.append(f"'img_size' {img_size} not recommended. Use one of [320, 416, 512, 640, 832]")
    
    return errors

def validate_runtime_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate runtime parameters for model operations
    
    Args:
        params: Runtime parameters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate confidence threshold
    if 'confidence_threshold' in params:
        conf = params['confidence_threshold']
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            return False, "Confidence threshold must be between 0 and 1"
    
    # Validate IOU threshold
    if 'iou_threshold' in params:
        iou = params['iou_threshold']
        if not isinstance(iou, (int, float)) or iou < 0 or iou > 1:
            return False, "IOU threshold must be between 0 and 1"
    
    # Validate max detections
    if 'max_detections' in params:
        max_det = params['max_detections']
        if not isinstance(max_det, int) or max_det < 1:
            return False, "Max detections must be a positive integer"
    
    return True, ""