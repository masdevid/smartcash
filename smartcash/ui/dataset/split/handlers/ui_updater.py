"""
File: smartcash/ui/dataset/split/handlers/ui_updater.py
Deskripsi: Update UI dari config - refactored dengan one-liner approach
"""

from typing import Dict, Any


def update_split_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan one-liner updates"""
    data_config = config.get('data', {})
    split_ratios = data_config.get('split_ratios', {})
    validation_config = data_config.get('validation', {})
    backup_config = config.get('dataset', {}).get('backup', {})
    
    # One-liner component update dengan safe access
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update ratio sliders
    [safe_update(f'{ratio}_slider', split_ratios.get(ratio, default)) for ratio, default in [('train', 0.7), ('valid', 0.15), ('test', 0.15)]]
    
    # Update form fields dengan mapping approach - sesuai dengan dataset_config.yaml
    field_mappings = [
        # Data validation settings
        ('stratified_checkbox', data_config, 'stratified_split', True),
        ('random_seed', data_config, 'random_seed', 42),
        
        # Validation settings
        ('validation_enabled', validation_config, 'enabled', True),
        ('fix_issues', validation_config, 'fix_issues', True),
        ('move_invalid', validation_config, 'move_invalid', True),
        ('invalid_dir', validation_config, 'invalid_dir', 'data/invalid'),
        ('visualize_issues', validation_config, 'visualize_issues', False),
        
        # Backup settings
        ('backup_checkbox', backup_config, 'enabled', True),
        ('backup_dir', backup_config, 'dir', 'data/backup/dataset'),
        ('backup_count', backup_config, 'count', 3),
        ('auto_backup', backup_config, 'auto', False)
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