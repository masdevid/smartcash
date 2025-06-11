"""
File: smartcash/dataset/preprocessor/config/validator.py
Deskripsi: Simplified config validation untuk API parameters
"""

from typing import Dict, Any, List
from .defaults import NORMALIZATION_PRESETS, get_default_config

def validate_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """✅ Validate dan merge config dengan defaults"""
    default = get_default_config()
    validated = _deep_merge(default, config or {})
    
    errors = []
    errors.extend(_validate_normalization(validated.get('preprocessing', {}).get('normalization', {})))
    errors.extend(_validate_splits(validated.get('preprocessing', {}).get('target_splits', [])))
    errors.extend(_validate_paths(validated.get('data', {})))
    
    if errors:
        raise ValueError(f"Config validation errors: {'; '.join(errors)}")
    
    return validated

def _validate_normalization(norm_config: Dict[str, Any]) -> List[str]:
    """🎯 Validate normalization parameters"""
    errors = []
    
    target_size = norm_config.get('target_size', [640, 640])
    if not isinstance(target_size, list) or len(target_size) != 2:
        errors.append("target_size must be [width, height]")
    elif not all(isinstance(x, int) and x > 0 for x in target_size):
        errors.append("target_size values must be positive integers")
    
    pixel_range = norm_config.get('pixel_range', [0, 1])
    if not isinstance(pixel_range, list) or len(pixel_range) != 2:
        errors.append("pixel_range must be [min, max]")
    elif pixel_range[0] >= pixel_range[1]:
        errors.append("pixel_range min must be less than max")
    
    interpolation = norm_config.get('interpolation', 'linear')
    if interpolation not in ['linear', 'nearest', 'cubic', 'lanczos']:
        errors.append(f"interpolation '{interpolation}' not supported")
    
    return errors

def _validate_splits(splits: List[str]) -> List[str]:
    """📂 Validate target splits"""
    errors = []
    valid_splits = ['train', 'valid', 'test', 'all']
    
    if isinstance(splits, str):
        splits = [splits]
    
    for split in splits:
        if split not in valid_splits:
            errors.append(f"Invalid split '{split}'. Valid: {valid_splits}")
    
    return errors

def _validate_paths(data_config: Dict[str, Any]) -> List[str]:
    """📁 Validate path configuration"""
    errors = []
    
    required_paths = ['dir', 'preprocessed_dir']
    for path_key in required_paths:
        if not data_config.get(path_key):
            errors.append(f"Missing required path: {path_key}")
    
    return errors

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """🔄 Deep merge configurations"""
    import copy
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def validate_normalization_preset(preset: str) -> str:
    """🎯 Validate normalization preset"""
    if preset not in NORMALIZATION_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(NORMALIZATION_PRESETS.keys())}")
    return preset

def get_validated_config(config: Dict[str, Any] = None, preset: str = 'default') -> Dict[str, Any]:
    """🔧 Get validated config dengan preset"""
    base_config = get_default_config()
    
    if preset != 'default':
        validate_normalization_preset(preset)
        base_config['preprocessing']['normalization'] = NORMALIZATION_PRESETS[preset].copy()
    
    return validate_preprocessing_config(base_config if config is None else config)