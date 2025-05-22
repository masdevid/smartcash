"""
File: smartcash/ui/dataset/split/handlers/ui_updater.py
Deskripsi: Update komponen UI dari konfigurasi split dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update komponen UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi untuk update UI
    """
    try:
        # Ekstrak data dari config
        split_ratios = config.get('data', {}).get('split_ratios', {})
        split_settings = config.get('split_settings', {})
        data_config = config.get('data', {})
        
        # Update slider ratio
        _update_ratio_sliders(ui_components, split_ratios)
        
        # Update checkbox dan input fields
        _update_form_fields(ui_components, data_config, split_settings)
        
        # Update total label
        _update_total_label(ui_components)
        
        logger.debug("🔄 UI berhasil diupdate dari konfigurasi")
        
    except Exception as e:
        logger.error(f"💥 Error updating UI from config: {str(e)}")


def _update_ratio_sliders(ui_components: Dict[str, Any], split_ratios: Dict[str, Any]) -> None:
    """Update slider ratio dari konfigurasi."""
    ratio_mapping = [
        ('train_slider', 'train', 0.7),
        ('valid_slider', 'valid', 0.15), 
        ('test_slider', 'test', 0.15)
    ]
    
    for slider_key, config_key, default_value in ratio_mapping:
        if slider_key in ui_components:
            ui_components[slider_key].value = split_ratios.get(config_key, default_value)


def _update_form_fields(ui_components: Dict[str, Any], data_config: Dict[str, Any], split_settings: Dict[str, Any]) -> None:
    """Update form fields dari konfigurasi."""
    field_mapping = [
        ('stratified_checkbox', data_config, 'stratified_split', True),
        ('random_seed', data_config, 'random_seed', 42),
        ('backup_checkbox', split_settings, 'backup_before_split', True),
        ('backup_dir', split_settings, 'backup_dir', 'data/splits_backup'),
        ('dataset_path', split_settings, 'dataset_path', 'data'),
        ('preprocessed_path', split_settings, 'preprocessed_path', 'data/preprocessed')
    ]
    
    for component_key, source_config, config_key, default_value in field_mapping:
        if component_key in ui_components:
            ui_components[component_key].value = source_config.get(config_key, default_value)


def _update_total_label(ui_components: Dict[str, Any]) -> None:
    """Update label total ratio."""
    try:
        if all(key in ui_components for key in ['train_slider', 'valid_slider', 'test_slider', 'total_label']):
            from smartcash.ui.utils.constants import COLORS
            
            total = round(
                ui_components['train_slider'].value + 
                ui_components['valid_slider'].value + 
                ui_components['test_slider'].value, 2
            )
            color = COLORS['success'] if total == 1.0 else COLORS['danger']
            ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
            
    except Exception as e:
        logger.warning(f"⚠️ Error updating total label: {str(e)}")


def reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke nilai default."""
    try:
        from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
        default_config = get_default_split_config()
        update_ui_from_config(ui_components, default_config)
        logger.info("🔄 UI berhasil direset ke default")
        
    except Exception as e:
        logger.error(f"💥 Error resetting UI to defaults: {str(e)}")