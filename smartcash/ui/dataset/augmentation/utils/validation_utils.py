"""
File: smartcash/ui/dataset/augmentation/utils/validation_utils.py
Deskripsi: Validation utilities untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, List, Union, Callable
import logging

logger = logging.getLogger(__name__)

def get_widget_value(ui_components: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safe widget value extraction dengan backend compatibility
    
    Args:
        ui_components: Dictionary berisi komponen UI
        key: Key untuk widget
        default: Default value jika widget tidak ditemukan
        
    Returns:
        Widget value atau default
    """
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = getattr(widget, 'value')
            
            # Enhanced type preservation dan validation
            if isinstance(default, bool) and not isinstance(value, bool):
                return bool(value)
            elif isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            elif isinstance(default, list) and not isinstance(value, list):
                return [value] if value else []
            elif isinstance(default, str) and not isinstance(value, str):
                return str(value) if value is not None else default
            
            return value
        except Exception:
            pass
    return default

def extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Enhanced augmentation types extraction dengan validation
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        List of augmentation types
    """
    types_widget = ui_components.get('augmentation_types')
    if types_widget and hasattr(types_widget, 'value'):
        try:
            value = getattr(types_widget, 'value')
            if isinstance(value, (list, tuple)) and value:
                # Validate tegen available types
                valid_types = ['combined', 'position', 'lighting', 'geometric', 'color', 'noise']
                filtered_types = [t for t in value if t in valid_types]
                return filtered_types if filtered_types else ['combined']
        except Exception:
            pass
    
    # Fallback ke combined jika tidak ada selection valid
    return ['combined']

def validate_augmentation_form(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced form validation dengan HSV parameters dan cleanup target
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary berisi hasil validasi
    """
    validation_result = {'valid': True, 'errors': [], 'warnings': [], 'backend_compatible': True}
    
    # Basic validation rules dengan enhanced ranges
    validations = [
        ('num_variations', lambda x: 1 <= x <= 10, "Jumlah variasi harus antara 1-10"),
        ('target_count', lambda x: 100 <= x <= 2000, "Target count harus antara 100-2000"),
        ('cleanup_target', lambda x: x in ['augmented', 'samples', 'both'], "Cleanup target harus valid"),
        ('augmentation_types', lambda x: x and len(x) > 0, "Pilih minimal 1 jenis augmentasi"),
        ('target_split', lambda x: x in ['train', 'valid', 'test'], "Target split harus train, valid, atau test")
    ]
    
    for key, validator, error_msg in validations:
        value = get_widget_value(ui_components, key)
        try:
            if not validator(value):
                validation_result['valid'] = False
                validation_result['backend_compatible'] = False
                validation_result['errors'].append(f"❌ {error_msg}")
        except Exception:
            validation_result['valid'] = False
            validation_result['backend_compatible'] = False
            validation_result['errors'].append(f"❌ Error validating {key}")
    
    # Advanced parameter validation dengan HSV
    _validate_position_parameters(ui_components, validation_result)
    _validate_lighting_parameters_with_hsv(ui_components, validation_result)
    _validate_cleanup_parameters(ui_components, validation_result)
    
    # Backend compatibility check
    if validation_result['valid']:
        _check_backend_compatibility(ui_components, validation_result)
    
    return validation_result

def _validate_position_parameters(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Validate position parameters dengan realistic ranges
    
    Args:
        ui_components: Dictionary berisi komponen UI
        validation_result: Dictionary berisi hasil validasi
    """
    fliplr = get_widget_value(ui_components, 'fliplr', 0.5)
    degrees = get_widget_value(ui_components, 'degrees', 12)
    translate = get_widget_value(ui_components, 'translate', 0.08)
    scale = get_widget_value(ui_components, 'scale', 0.04)
    
    if fliplr > 0.8:
        validation_result['warnings'].append("⚠️ Flip probability sangat tinggi (>80%) - hasil mungkin tidak natural")
    
    if degrees > 20:
        validation_result['warnings'].append("⚠️ Rotasi >20° mungkin terlalu ekstrem untuk uang kertas")
    
    if translate > 0.15 or scale > 0.15:
        validation_result['warnings'].append("⚠️ Translate/Scale >15% mungkin mengubah proporsi terlalu drastis")

def _validate_lighting_parameters_with_hsv(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Validate lighting parameters dengan HSV support
    
    Args:
        ui_components: Dictionary berisi komponen UI
        validation_result: Dictionary berisi hasil validasi
    """
    brightness = get_widget_value(ui_components, 'brightness', 0.2)
    contrast = get_widget_value(ui_components, 'contrast', 0.15)
    hsv_h = get_widget_value(ui_components, 'hsv_h', 10)
    hsv_s = get_widget_value(ui_components, 'hsv_s', 15)
    
    if brightness > 0.3 or contrast > 0.3:
        validation_result['warnings'].append("⚠️ Brightness/Contrast >30% mungkin menghasilkan gambar tidak realistis")
    
    if brightness < 0.05 and contrast < 0.05:
        validation_result['warnings'].append("⚠️ Variasi pencahayaan sangat rendah - augmentasi mungkin tidak efektif")
    
    # HSV validation
    if hsv_h > 25:
        validation_result['warnings'].append("⚠️ HSV Hue >25 mungkin mengubah warna terlalu drastis")
    
    if hsv_s > 40:
        validation_result['warnings'].append("⚠️ HSV Saturation >40 mungkin menghasilkan warna tidak natural")

def _validate_cleanup_parameters(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Validate cleanup parameters
    
    Args:
        ui_components: Dictionary berisi komponen UI
        validation_result: Dictionary berisi hasil validasi
    """
    cleanup_target = get_widget_value(ui_components, 'cleanup_target', 'both')
    target_split = get_widget_value(ui_components, 'target_split', 'train')
    
    if cleanup_target == 'both':
        validation_result['warnings'].append("⚠️ Cleanup 'both' akan menghapus semua file augmented dan samples")
    
    if target_split == 'test' and cleanup_target in ['augmented', 'both']:
        validation_result['warnings'].append("⚠️ Cleanup pada test split tidak direkomendasikan")

def _check_backend_compatibility(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Check backend service compatibility
    
    Args:
        ui_components: Dictionary berisi komponen UI
        validation_result: Dictionary berisi hasil validasi
    """
    try:
        backend_ready = ui_components.get('backend_ready', False)
        service_integration = ui_components.get('service_integration', False)
        
        if not (backend_ready and service_integration):
            validation_result['warnings'].append("⚠️ Backend service tidak sepenuhnya terintegrasi")
            validation_result['backend_compatible'] = False
    except Exception:
        validation_result['backend_compatible'] = False
