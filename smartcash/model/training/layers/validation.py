"""
Layer validation and configuration utilities.

This module provides functions for validating layer configurations and parameters
used throughout the training pipeline.
"""

from typing import Dict, List, Tuple, Any
from .constants import LAYER_CONFIG, DETECTION_LAYERS

# One-liner utilities for common operations
get_primary_layer = lambda detection_layers: detection_layers[0] if detection_layers else 'banknote'
get_total_classes = lambda detection_layers: sum(
    LAYER_CONFIG.get(layer, {}).get('num_classes', 0) 
    for layer in detection_layers
)
is_multilayer_capable = lambda detection_layers: len([
    l for l in detection_layers if l in DETECTION_LAYERS
]) >= 2
get_valid_layers_only = lambda detection_layers: [
    l for l in detection_layers if l in DETECTION_LAYERS
]

def validate_layer_params(layer_mode: str, detection_layers: List[str]) -> Tuple[str, List[str]]:
    """
    Validate layer parameters for training configuration.
    
    Args:
        layer_mode: Layer mode ('single' or 'multilayer')
        detection_layers: List of detection layers to validate
        
    Returns:
        Tuple of (validated_layer_mode, validated_detection_layers)
    """
    # Filter and validate detection layers
    valid_layers = get_valid_layers_only(detection_layers)
    
    # If no valid layers, use default
    if not valid_layers:
        valid_layers = ['banknote']
    
    # Auto-correct mode based on number of layers
    if layer_mode == 'multilayer' and len(valid_layers) < 2:
        corrected_mode = 'single'
    elif layer_mode == 'single' and len(valid_layers) >= 2:
        # Keep as single if user explicitly chose single
        corrected_mode = 'single'
    else:
        corrected_mode = layer_mode
    
    return corrected_mode, valid_layers

def is_valid_layer_combination(layer_mode: str, detection_layers: List[str]) -> bool:
    """
    Check if the combination of layer mode and detection layers is valid.
    
    Args:
        layer_mode: Layer mode ('single' or 'multilayer')
        detection_layers: List of detection layers
        
    Returns:
        bool: True if the combination is valid, False otherwise
    """
    valid_layers = get_valid_layers_only(detection_layers)
    
    if layer_mode == 'multilayer':
        return len(valid_layers) >= 2
    elif layer_mode == 'single':
        return len(valid_layers) >= 1
    
    return False

def auto_determine_layer_mode(detection_layers: List[str]) -> str:
    """
    Automatically determine the appropriate layer mode based on detection layers.
    
    Args:
        detection_layers: List of detection layers
        
    Returns:
        str: 'single' or 'multilayer'
    """
    valid_layers = get_valid_layers_only(detection_layers)
    return 'multilayer' if len(valid_layers) >= 2 else 'single'

def get_layer_info(layer_name: str) -> Dict[str, Any]:
    """
    Get configuration information for a specific layer.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Dictionary containing layer configuration
    """
    return LAYER_CONFIG.get(layer_name, {})

def get_layer_offsets(detection_layers: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Calculate class index offsets for each layer in multi-layer mode.
    
    Args:
        detection_layers: List of detection layers
        
    Returns:
        Dictionary mapping layer names to (start_idx, end_idx) tuples
    """
    offsets = {}
    current_offset = 0
    
    for layer in detection_layers:
        if layer in LAYER_CONFIG:
            num_classes = LAYER_CONFIG[layer].get('num_classes', 0)
            if num_classes > 0:
                offsets[layer] = (current_offset, current_offset + num_classes)
                current_offset += num_classes
    
    return offsets

def get_num_classes_for_layers(detection_layers: List[str]) -> int:
    """
    Calculate the total number of classes across all detection layers.
    
    Args:
        detection_layers: List of detection layers
        
    Returns:
        Total number of classes
    """
    return sum(
        LAYER_CONFIG[layer].get('num_classes', 0)
        for layer in detection_layers
        if layer in LAYER_CONFIG
    )
