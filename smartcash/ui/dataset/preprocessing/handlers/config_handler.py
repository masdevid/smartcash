"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler untuk konfigurasi preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components berdasarkan konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
    """
    # Get preprocessing config
    preprocessing_config = config.get('preprocessing', {})
    if not preprocessing_config:
        return
    
    # Update preprocessing options
    if 'preprocess_options' in ui_components:
        options = ui_components['preprocess_options']
        
        # Resolution
        if 'resolution' in preprocessing_config:
            resolution = preprocessing_config['resolution']
            if hasattr(options, 'resolution') and hasattr(options.resolution, 'value'):
                # Convert tuple to string if needed
                if isinstance(resolution, tuple) and len(resolution) == 2:
                    width, height = resolution
                    resolution_str = f"{width}x{height}"
                    options.resolution.value = resolution_str
                elif isinstance(resolution, str):
                    options.resolution.value = resolution
                
        # Normalization
        if 'normalization' in preprocessing_config:
            normalization = preprocessing_config['normalization']
            if hasattr(options, 'normalization') and hasattr(options.normalization, 'value'):
                options.normalization.value = normalization
                
        # Augmentation
        if 'augmentation' in preprocessing_config:
            augmentation = preprocessing_config['augmentation']
            if hasattr(options, 'augmentation') and hasattr(options.augmentation, 'value'):
                options.augmentation.value = bool(augmentation)
    
    # Update split selector
    if 'split' in preprocessing_config and 'split_selector' in ui_components:
        split_value = preprocessing_config['split']
        split_map = {
            'train': 'Train Only',
            'val': 'Validation Only',
            'test': 'Test Only',
            'all': 'All Splits'
        }
        if hasattr(ui_components['split_selector'], 'value'):
            if split_value in split_map:
                ui_components['split_selector'].value = split_map[split_value]
    
    # Update validation options
    if 'validation_items' in preprocessing_config and 'validation_options' in ui_components:
        validation_items = preprocessing_config['validation_items']
        if hasattr(ui_components['validation_options'], 'set_selected') and callable(ui_components['validation_options'].set_selected):
            try:
                ui_components['validation_options'].set_selected(validation_items)
            except (AttributeError, TypeError):
                # Ignore errors jika method tidak tersedia
                pass
    
    # Log
    log_message(ui_components, "UI diperbarui dari konfigurasi", "info", "🔄") 