"""
File: smartcash/dataset/augmentor/config.py
Deskripsi: Updated config menggunakan SRP modules
"""

from typing import Dict, Any, List

# Updated imports dari SRP modules - menggantikan core imports
from smartcash.dataset.augmentor.utils.path_operations import resolve_drive_path, get_best_data_location
from smartcash.dataset.augmentor.utils.config_extractor import extract_config

from .types import AugConfig, DEFAULT_AUGMENTATION_TYPES

# One-liner config extractors dengan aligned parameters menggunakan SRP modules
get_raw_dir = lambda cfg: resolve_drive_path(cfg.get('data', {}).get('dir', 'data'))
get_aug_dir = lambda cfg: resolve_drive_path(cfg.get('augmentation', {}).get('output_dir', 'data/augmented'))
get_prep_dir = lambda cfg: resolve_drive_path(cfg.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))

# Parameter extractors dengan UI alignment
get_num_variations = lambda cfg: cfg.get('augmentation', {}).get('num_variations', 2)
get_target_count = lambda cfg: cfg.get('augmentation', {}).get('target_count', 500)
get_augmentation_types = lambda cfg: cfg.get('augmentation', {}).get('types', DEFAULT_AUGMENTATION_TYPES)
get_intensity = lambda cfg: cfg.get('augmentation', {}).get('intensity', 0.7)
get_target_split = lambda cfg: cfg.get('augmentation', {}).get('target_split', 'train')

# UI parameter extractors
get_fliplr = lambda cfg: cfg.get('augmentation', {}).get('fliplr', 0.5)
get_degrees = lambda cfg: cfg.get('augmentation', {}).get('degrees', 10)
get_translate = lambda cfg: cfg.get('augmentation', {}).get('translate', 0.1)
get_scale = lambda cfg: cfg.get('augmentation', {}).get('scale', 0.1)
get_hsv_h = lambda cfg: cfg.get('augmentation', {}).get('hsv_h', 0.015)
get_hsv_s = lambda cfg: cfg.get('augmentation', {}).get('hsv_s', 0.7)
get_brightness = lambda cfg: cfg.get('augmentation', {}).get('brightness', 0.2)
get_contrast = lambda cfg: cfg.get('augmentation', {}).get('contrast', 0.2)

# One-liner validators menggunakan SRP functions
validate_config = lambda cfg: cfg and isinstance(cfg, dict)
validate_paths = lambda cfg: all([get_raw_dir(cfg), get_aug_dir(cfg), get_prep_dir(cfg)])

def create_aug_config(config: Dict[str, Any]) -> AugConfig:
    """Create AugConfig dengan aligned parameters menggunakan SRP modules"""
    if not validate_config(config):
        return AugConfig()
    
    return AugConfig(
        raw_dir=get_raw_dir(config),
        aug_dir=get_aug_dir(config),
        prep_dir=get_prep_dir(config),
        num_variations=max(1, get_num_variations(config)),
        target_count=max(1, get_target_count(config)),
        validate_results=False
    )

def extract_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI dengan parameter alignment menggunakan SRP modules"""
    config = ui_components.get('config', {})
    base_data_dir = get_best_data_location()  # Menggunakan SRP function
    
    # Extract augmentation types dengan fallback strategies
    selected_types = _extract_augmentation_types_safe(ui_components)
    
    # Extract parameters menggunakan widget extraction
    num_variations = _get_widget_value_safe(ui_components, 'num_variations', 2)
    target_count = _get_widget_value_safe(ui_components, 'target_count', 500)
    target_split = _get_widget_value_safe(ui_components, 'target_split', 'train')
    
    # Advanced parameters
    advanced_params = {
        'fliplr': _get_widget_value_safe(ui_components, 'fliplr', 0.5),
        'degrees': _get_widget_value_safe(ui_components, 'degrees', 10),
        'translate': _get_widget_value_safe(ui_components, 'translate', 0.1),
        'scale': _get_widget_value_safe(ui_components, 'scale', 0.1),
        'hsv_h': _get_widget_value_safe(ui_components, 'hsv_h', 0.015),
        'hsv_s': _get_widget_value_safe(ui_components, 'hsv_s', 0.7),
        'brightness': _get_widget_value_safe(ui_components, 'brightness', 0.2),
        'contrast': _get_widget_value_safe(ui_components, 'contrast', 0.2)
    }
    
    return {
        'data': {'dir': base_data_dir},
        'augmentation': {
            'types': selected_types, 'num_variations': num_variations,
            'target_count': target_count, 'target_split': target_split,
            'intensity': config.get('intensity', 0.7),
            'output_dir': resolve_drive_path('data/augmented'),  # SRP function
            **advanced_params
        },
        'preprocessing': {'output_dir': resolve_drive_path('data/preprocessed')}  # SRP function
    }

def _extract_augmentation_types_safe(ui_components: Dict[str, Any]) -> List[str]:
    """Safe extraction dengan multiple fallback strategies"""
    # Strategy 1: Direct widget
    aug_types_widget = ui_components.get('augmentation_types')
    if aug_types_widget and hasattr(aug_types_widget, 'value') and aug_types_widget.value:
        return list(aug_types_widget.value)
    
    # Strategy 2: Container widget
    aug_options = ui_components.get('aug_options')
    if aug_options:
        if hasattr(aug_options, 'children') and aug_options.children:
            first_child = aug_options.children[0]
            if hasattr(first_child, 'value') and first_child.value:
                return list(first_child.value)
        elif hasattr(aug_options, 'value') and aug_options.value:
            return list(aug_options.value)
    
    # Strategy 3: Alternative widget name
    types_widget = ui_components.get('types_widget')
    if types_widget and hasattr(types_widget, 'value') and types_widget.value:
        return list(types_widget.value)
    
    # Strategy 4: Config fallback
    config = ui_components.get('config', {})
    aug_config = config.get('augmentation', {})
    if 'types' in aug_config:
        return list(aug_config['types'])
    
    return ['combined']  # Default

def _get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Safe widget value extraction dengan type checking"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = widget.value
            # Type consistency check
            if isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            elif isinstance(default, str) and hasattr(value, '__str__'):
                return str(value)
            else:
                return value
        except Exception:
            pass
    return default

def apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Apply config ke UI widgets dengan parameter alignment"""
    aug_config = config.get('augmentation', {})
    
    # Basic parameters
    basic_mappings = {
        'num_variations': aug_config.get('num_variations', 2),
        'target_count': aug_config.get('target_count', 500),
        'target_split': aug_config.get('target_split', 'train')
    }
    
    # Advanced parameters
    advanced_mappings = {
        'fliplr': aug_config.get('fliplr', 0.5),
        'degrees': aug_config.get('degrees', 10),
        'translate': aug_config.get('translate', 0.1),
        'scale': aug_config.get('scale', 0.1),
        'hsv_h': aug_config.get('hsv_h', 0.015),
        'hsv_s': aug_config.get('hsv_s', 0.7),
        'brightness': aug_config.get('brightness', 0.2),
        'contrast': aug_config.get('contrast', 0.2)
    }
    
    # Apply mappings
    all_mappings = {**basic_mappings, **advanced_mappings}
    for widget_key, value in all_mappings.items():
        _set_widget_value_safe(ui_components, widget_key, value)
    
    # Apply augmentation types
    aug_types = aug_config.get('types', ['combined'])
    _set_augmentation_types_safe(ui_components, aug_types)

def _set_widget_value_safe(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safe widget value setting"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            widget.value = value
        except Exception:
            pass

def _set_augmentation_types_safe(ui_components: Dict[str, Any], types: List[str]) -> None:
    """Safe augmentation types setting"""
    for widget_key in ['augmentation_types', 'types_widget']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = list(types)
                return
            except Exception:
                continue
    
    # Try container
    aug_options = ui_components.get('aug_options')
    if aug_options and hasattr(aug_options, 'children') and aug_options.children:
        try:
            aug_options.children[0].value = list(types)
        except Exception:
            pass

# Auto-detect config menggunakan SRP modules
auto_detect_config = lambda: {
    'data': {'dir': get_best_data_location()},
    'augmentation': {
        'types': DEFAULT_AUGMENTATION_TYPES, 'num_variations': 2, 'target_count': 500,
        'intensity': 0.7, 'target_split': 'train', 'output_dir': resolve_drive_path('data/augmented'),
        'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
        'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2
    },
    'preprocessing': {'output_dir': resolve_drive_path('data/preprocessed')}
}

# One-liner utilities dengan SRP integration
merge_configs = lambda base, override: {**base, **{k: {**base.get(k, {}), **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in override.items()}}
normalize_config = lambda cfg: {k: v for k, v in cfg.items() if v is not None}
validate_ui_parameters = lambda ui_components: all([_get_widget_value_safe(ui_components, 'num_variations', 0) > 0, _get_widget_value_safe(ui_components, 'target_count', 0) > 0, len(_extract_augmentation_types_safe(ui_components)) > 0])

get_ui_parameter_summary = lambda ui_components: {
    'types': _extract_augmentation_types_safe(ui_components),
    'num_variations': _get_widget_value_safe(ui_components, 'num_variations', 2),
    'target_count': _get_widget_value_safe(ui_components, 'target_count', 500),
    'target_split': _get_widget_value_safe(ui_components, 'target_split', 'train'),
    'intensity_params': {
        'degrees': _get_widget_value_safe(ui_components, 'degrees', 10),
        'brightness': _get_widget_value_safe(ui_components, 'brightness', 0.2),
        'contrast': _get_widget_value_safe(ui_components, 'contrast', 0.2)
    }
}