"""
File: smartcash/ui/dataset/split/handlers/config_updater.py
Deskripsi: Update UI dari config dengan centralized error handling
"""

from typing import Dict, Any, List, Tuple
import logging

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors

# Import constants
from smartcash.ui.utils.constants import COLORS

# Import defaults
from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config

# Logger
logger = logging.getLogger(__name__)


@handle_ui_errors(log_error=True)
def update_split_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan centralized error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi split dataset
    """
    logger.debug("Memperbarui UI split dari konfigurasi")
    
    # Extract nested configs dengan fallbacks
    data_config = config.get('data', {})
    split_ratios = data_config.get('split_ratios', {})
    validation_config = data_config.get('validation', {})
    backup_config = config.get('dataset', {}).get('backup', {})
    split_settings = config.get('split_settings', {})
    
    # Update ratio sliders
    _update_ratio_sliders(ui_components, split_ratios)
    
    # Update form fields
    _update_form_fields(ui_components, data_config, validation_config, backup_config, split_settings)
    
    # Update total label
    _update_total_label_safe(ui_components)
    
    logger.debug("UI split berhasil diperbarui dari konfigurasi")


@handle_ui_errors(log_error=True)
def _update_ratio_sliders(ui_components: Dict[str, Any], split_ratios: Dict[str, Any]) -> None:
    """Update ratio sliders dari konfigurasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        split_ratios: Dictionary berisi nilai split ratios
    """
    # Default values
    defaults = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    
    # Update each slider
    for ratio, default in defaults.items():
        slider_key = f'{ratio}_slider'
        if slider_key in ui_components and hasattr(ui_components[slider_key], 'value'):
            ui_components[slider_key].value = split_ratios.get(ratio, default)
            logger.debug(f"Slider {ratio} diperbarui ke {split_ratios.get(ratio, default)}")
        else:
            logger.warning(f"Slider {ratio} tidak ditemukan dalam UI components")


@handle_ui_errors(log_error=True)
def _update_form_fields(
    ui_components: Dict[str, Any], 
    data_config: Dict[str, Any],
    validation_config: Dict[str, Any],
    backup_config: Dict[str, Any],
    split_settings: Dict[str, Any]
) -> None:
    """Update form fields dari konfigurasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        data_config: Dictionary konfigurasi data
        validation_config: Dictionary konfigurasi validasi
        backup_config: Dictionary konfigurasi backup
        split_settings: Dictionary konfigurasi split settings
    """
    # Define field mappings
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
        
        # Backup settings - dari dataset.backup
        ('backup_checkbox', backup_config, 'enabled', True),
        ('backup_dir', backup_config, 'dir', 'data/backup/dataset'),
        ('backup_count', backup_config, 'count', 2),
        ('auto_backup', backup_config, 'auto', False),
        
        # Split settings - untuk backward compatibility
        ('backup_checkbox', split_settings, 'backup_before_split', True)
    ]
    
    # Update each field
    for component_key, source_config, config_key, default_value in field_mappings:
        if component_key in ui_components and hasattr(ui_components[component_key], 'value'):
            ui_components[component_key].value = source_config.get(config_key, default_value)
            logger.debug(f"Field {component_key} diperbarui ke {source_config.get(config_key, default_value)}")


@handle_ui_errors(log_error=True)
def _update_total_label_safe(ui_components: Dict[str, Any]) -> None:
    """Update total label dengan centralized error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    if 'total_label' not in ui_components:
        logger.warning("Total label tidak ditemukan dalam UI components")
        return
    
    # Check if all sliders exist
    sliders = ['train_slider', 'valid_slider', 'test_slider']
    if not all(slider in ui_components for slider in sliders):
        logger.warning("Tidak semua slider ditemukan dalam UI components")
        return
    
    # Calculate total
    total = round(sum(getattr(ui_components[f'{ratio}_slider'], 'value', 0) for ratio in ['train', 'valid', 'test']), 2)
    
    # Determine color based on total
    is_valid = total == 1.0
    color = COLORS.get('success', '#28a745') if is_valid else COLORS.get('danger', '#dc3545')
    
    # Update label
    ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
    
    # Update save button state if available
    if 'save_button' in ui_components:
        ui_components['save_button'].disabled = not is_valid
    
    # Log result
    if is_valid:
        logger.debug(f"Total ratio valid: {total}")
    else:
        logger.warning(f"Total ratio tidak valid: {total}, seharusnya 1.0")


@handle_ui_errors(log_error=True)
def reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke nilai default
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger.info("Mereset UI split ke nilai default")
    update_split_ui(ui_components, get_default_split_config())
    logger.info("UI split berhasil direset ke nilai default")


@handle_ui_errors(log_error=True)
def get_current_ratios(ui_components: Dict[str, Any]) -> Dict[str, float]:
    """Get current ratio values dari UI components
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary berisi nilai ratio saat ini
    """
    defaults = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    result = {}
    
    for ratio, default in defaults.items():
        slider = ui_components.get(f'{ratio}_slider')
        if slider is None:
            logger.warning(f"{ratio}_slider tidak ditemukan, menggunakan nilai default {default}")
            result[ratio] = default
        else:
            result[ratio] = getattr(slider, 'value', default)
    
    return result
