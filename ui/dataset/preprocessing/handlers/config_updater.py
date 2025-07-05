"""
File: smartcash/ui/dataset/preprocessing/handlers/config_updater.py
Deskripsi: UI updater untuk essential preprocessing forms dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple, Union
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.decorators.ui_decorators import safe_ui_operation

@handle_ui_errors(error_component_title="UI Update Error", log_error=True)
def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dengan config values.
    
    Args:
        ui_components: Dictionary containing UI components
        config: Configuration dictionary
    """
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    validation_config = preprocessing_config.get('validation', {})
    cleanup_config = preprocessing_config.get('cleanup', {})
    performance_config = config.get('performance', {})
    
    # Update UI components
    _update_resolution(ui_components, normalization_config)
    _update_normalization(ui_components, normalization_config)
    _update_target_splits(ui_components, preprocessing_config)
    _update_validation_settings(ui_components, validation_config)
    _update_cleanup_settings(ui_components, cleanup_config)
    _update_performance_settings(ui_components, performance_config)

@safe_ui_operation(error_title="UI Reset Error")
def reset_preprocessing_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default values.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    try:
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        default_config = get_default_preprocessing_config()
        update_preprocessing_ui(ui_components, default_config)
    except Exception:
        _apply_hardcoded_defaults(ui_components)

@safe_ui_operation(error_title="UI Default Apply Error")
def _apply_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Hardcoded defaults fallback.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    defaults = {
        'resolution_dropdown': '640x640',
        'normalization_dropdown': 'minmax',
        'target_splits_select': ('train', 'valid'),
        'preserve_aspect_checkbox': True,
        'validation_checkbox': True,
        'move_invalid_checkbox': True,
        'invalid_dir_input': 'data/invalid',
        'cleanup_target_dropdown': 'preprocessed',
        'backup_checkbox': False,
        'batch_size_input': 32
    }
    
    _update_ui_components(ui_components, defaults)

@safe_ui_operation(error_title="UI Component Update Error")
def _update_ui_components(ui_components: Dict[str, Any], values: Dict[str, Any]) -> None:
    """Update UI components dengan values.
    
    Args:
        ui_components: Dictionary containing UI components
        values: Dictionary of values to update
    """
    for key, value in values.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass

@safe_ui_operation(error_title="Resolution Update Error")
def _update_resolution(ui_components: Dict[str, Any], normalization_config: Dict[str, Any]) -> None:
    """Update resolution dropdown.
    
    Args:
        ui_components: Dictionary containing UI components
        normalization_config: Normalization configuration
    """
    try:
        target_size = normalization_config.get('target_size', [640, 640])
        resolution_str = f"{target_size[0]}x{target_size[1]}"
        valid_resolutions = ['320x320', '416x416', '512x512', '640x640', '832x832']
        _safe_update(ui_components, 'resolution_dropdown', 
                    resolution_str if resolution_str in valid_resolutions else '640x640')
    except Exception:
        _safe_update(ui_components, 'resolution_dropdown', '640x640')

@safe_ui_operation(error_title="Normalization Update Error")
def _update_normalization(ui_components: Dict[str, Any], normalization_config: Dict[str, Any]) -> None:
    """Update normalization settings.
    
    Args:
        ui_components: Dictionary containing UI components
        normalization_config: Normalization configuration
    """
    try:
        normalization_enabled = normalization_config.get('enabled', True)
        method = normalization_config.get('method', 'minmax')
        final_value = 'none' if not normalization_enabled else method
        _safe_update(ui_components, 'normalization_dropdown', 
                    final_value if final_value in ['minmax', 'standard', 'none'] else 'minmax')
        _safe_update(ui_components, 'preserve_aspect_checkbox', 
                    bool(normalization_config.get('preserve_aspect_ratio', True)))
    except Exception:
        _safe_update(ui_components, 'normalization_dropdown', 'minmax')
        _safe_update(ui_components, 'preserve_aspect_checkbox', True)

@safe_ui_operation(error_title="Target Splits Update Error")
def _update_target_splits(ui_components: Dict[str, Any], preprocessing_config: Dict[str, Any]) -> None:
    """Update target splits.
    
    Args:
        ui_components: Dictionary containing UI components
        preprocessing_config: Preprocessing configuration
    """
    try:
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, list):
            validated_splits = [s for s in target_splits if s in ['train', 'valid', 'test']]
            _safe_update(ui_components, 'target_splits_select', 
                        tuple(validated_splits) if validated_splits else ('train', 'valid'))
        else:
            _safe_update(ui_components, 'target_splits_select', ('train', 'valid'))
    except Exception:
        _safe_update(ui_components, 'target_splits_select', ('train', 'valid'))

@safe_ui_operation(error_title="Validation Settings Update Error")
def _update_validation_settings(ui_components: Dict[str, Any], validation_config: Dict[str, Any]) -> None:
    """Update validation settings.
    
    Args:
        ui_components: Dictionary containing UI components
        validation_config: Validation configuration
    """
    try:
        _safe_update(ui_components, 'validation_checkbox', bool(validation_config.get('enabled', True)))
        _safe_update(ui_components, 'move_invalid_checkbox', bool(validation_config.get('move_invalid', True)))
        _safe_update(ui_components, 'invalid_dir_input', str(validation_config.get('invalid_dir', 'data/invalid')))
    except Exception:
        _safe_update(ui_components, 'validation_checkbox', True)
        _safe_update(ui_components, 'move_invalid_checkbox', True)
        _safe_update(ui_components, 'invalid_dir_input', 'data/invalid')

@safe_ui_operation(error_title="Cleanup Settings Update Error")
def _update_cleanup_settings(ui_components: Dict[str, Any], cleanup_config: Dict[str, Any]) -> None:
    """Update cleanup settings.
    
    Args:
        ui_components: Dictionary containing UI components
        cleanup_config: Cleanup configuration
    """
    try:
        cleanup_target = cleanup_config.get('target', 'preprocessed')
        valid_targets = ['preprocessed', 'samples', 'both']
        _safe_update(ui_components, 'cleanup_target_dropdown', 
                    cleanup_target if cleanup_target in valid_targets else 'preprocessed')
        _safe_update(ui_components, 'backup_checkbox', bool(cleanup_config.get('backup_enabled', False)))
    except Exception:
        _safe_update(ui_components, 'cleanup_target_dropdown', 'preprocessed')
        _safe_update(ui_components, 'backup_checkbox', False)

@safe_ui_operation(error_title="Performance Settings Update Error")
def _update_performance_settings(ui_components: Dict[str, Any], performance_config: Dict[str, Any]) -> None:
    """Update performance settings.
    
    Args:
        ui_components: Dictionary containing UI components
        performance_config: Performance configuration
    """
    try:
        batch_size = performance_config.get('batch_size', 32)
        _safe_update(ui_components, 'batch_size_input', 
                    max(1, min(batch_size, 128)) if isinstance(batch_size, int) else 32)
    except Exception:
        _safe_update(ui_components, 'batch_size_input', 32)

def _safe_update(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safely update UI component value.
    
    Args:
        ui_components: Dictionary containing UI components
        key: Component key
        value: New value
    """
    if key in ui_components and hasattr(ui_components[key], 'value'):
        try:
            ui_components[key].value = value
        except Exception:
            pass
