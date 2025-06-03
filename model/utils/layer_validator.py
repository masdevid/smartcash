"""
File: smartcash/model/utils/layer_validator.py
Deskripsi: Shared layer validation utilities untuk training UI dan model manager
"""

from typing import List, Tuple, Dict, Any
from smartcash.model.config.model_constants import LAYER_CONFIG, DETECTION_LAYERS

def validate_layer_params(layer_mode: str, detection_layers: List[str]) -> Tuple[str, List[str]]:
    """
    Unified layer parameter validation untuk training UI dan model manager.
    
    Args:
        layer_mode: Mode layer ('single' atau 'multilayer')
        detection_layers: List layer deteksi
        
    Returns:
        Tuple berisi (validated_layer_mode, validated_detection_layers)
    """
    # Validate dan filter detection layers
    valid_layers = [layer for layer in detection_layers if layer in DETECTION_LAYERS]
    
    # Jika tidak ada layer valid, gunakan default
    if not valid_layers: valid_layers = ['banknote']
    
    # Auto-correct mode berdasarkan jumlah layer
    if layer_mode == 'multilayer' and len(valid_layers) < 2:
        # Multilayer membutuhkan minimal 2 layer
        corrected_mode = 'single'
    elif layer_mode == 'single' and len(valid_layers) >= 2:
        # Keep as single jika user explicitly choose single
        corrected_mode = 'single'
    else:
        corrected_mode = layer_mode
    
    return corrected_mode, valid_layers

def get_num_classes_for_layers(detection_layers: List[str]) -> int:
    """Get total number classes untuk multilayer mode."""
    return sum(LAYER_CONFIG[layer]['num_classes'] for layer in detection_layers if layer in LAYER_CONFIG)

def get_layer_info(layer_name: str) -> Dict[str, Any]:
    """Get layer configuration info."""
    return LAYER_CONFIG.get(layer_name, {})

def is_valid_layer_combination(layer_mode: str, detection_layers: List[str]) -> bool:
    """Check apakah kombinasi layer mode dan detection layers valid."""
    valid_layers = [layer for layer in detection_layers if layer in DETECTION_LAYERS]
    
    if layer_mode == 'multilayer':
        return len(valid_layers) >= 2
    elif layer_mode == 'single':
        return len(valid_layers) >= 1
    
    return False

def get_layer_offsets(detection_layers: List[str]) -> Dict[str, Tuple[int, int]]:
    """Get class offset ranges untuk multilayer mode."""
    offsets = {}
    offset = 0
    
    for layer in detection_layers:
        if layer in LAYER_CONFIG:
            layer_classes = LAYER_CONFIG[layer]['num_classes']
            offsets[layer] = (offset, offset + layer_classes)
            offset += layer_classes
    
    return offsets

def auto_determine_layer_mode(detection_layers: List[str]) -> str:
    """Auto-determine layer mode berdasarkan detection layers."""
    valid_layers = [layer for layer in detection_layers if layer in DETECTION_LAYERS]
    return 'multilayer' if len(valid_layers) >= 2 else 'single'

# One-liner utilities
get_primary_layer = lambda detection_layers: detection_layers[0] if detection_layers else 'banknote'
get_total_classes = lambda detection_layers: sum(LAYER_CONFIG.get(layer, {}).get('num_classes', 0) for layer in detection_layers)
is_multilayer_capable = lambda detection_layers: len([l for l in detection_layers if l in DETECTION_LAYERS]) >= 2
get_valid_layers_only = lambda detection_layers: [l for l in detection_layers if l in DETECTION_LAYERS]