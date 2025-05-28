"""
File: smartcash/ui/dataset/split/handlers/ui_updater.py
Deskripsi: Update UI dari config - refactored dengan one-liner approach
"""

from typing import Dict, Any


def update_split_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan one-liner updates"""
    split_ratios = config.get('data', {}).get('split_ratios', {})
    split_settings = config.get('split_settings', {})
    data_config = config.get('data', {})
    
    # One-liner component update dengan safe access
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update ratio sliders
    [safe_update(f'{ratio}_slider', split_ratios.get(ratio, default)) for ratio, default in [('train', 0.7), ('valid', 0.15), ('test', 0.15)]]
    
    # Update form fields dengan mapping approach
    field_mappings = [
        ('stratified_checkbox', data_config, 'stratified_split', True),
        ('random_seed', data_config, 'random_seed', 42),
        ('backup_checkbox', split_settings, 'backup_before_split', True),
        ('backup_dir', split_settings, 'backup_dir', 'data/splits_backup'),
        ('dataset_path', split_settings, 'dataset_path', 'data'),
        ('preprocessed_path', split_settings, 'preprocessed_path', 'data/preprocessed')
    ]
    
    [safe_update(component_key, source_config.get(config_key, default_value)) for component_key, source_config, config_key, default_value in field_mappings]
    
    # Update total label with one-liner calculation
    _update_total_label_safe(ui_components)


def _update_total_label_safe(ui_components: Dict[str, Any]) -> None:
    """Update total label dengan safe access dan styling"""
    if not all(key in ui_components for key in ['train_slider', 'valid_slider', 'test_slider', 'total_label']):
        return
    
    try:
        from smartcash.ui.utils.constants import COLORS
        total = round(sum(getattr(ui_components[f'{ratio}_slider'], 'value', 0) for ratio in ['train', 'valid', 'test']), 2)
        color = COLORS.get('success', '#28a745') if total == 1.0 else COLORS.get('danger', '#dc3545')
        ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
    except Exception:
        pass  # Silent fail untuk prevent errors


# One-liner utility functions
reset_ui_to_defaults = lambda ui_components: update_split_ui(ui_components, __import__('smartcash.ui.dataset.split.handlers.defaults', fromlist=['get_default_split_config']).get_default_split_config())
get_current_ratios = lambda ui_components: {ratio: getattr(ui_components.get(f'{ratio}_slider', type('', (), {'value': default})()), 'value', default) for ratio, default in [('train', 0.7), ('valid', 0.15), ('test', 0.15)]}