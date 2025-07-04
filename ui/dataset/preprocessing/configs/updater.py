"""
File: smartcash/ui/dataset/preprocessing/configs/updater.py
Deskripsi: Config update utilities untuk preprocessing module.
"""

from typing import Dict, Any, List
import logging


def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update preprocessing UI components dari config.
    
    Args:
        ui_components: Dictionary containing UI components
        config: Configuration dictionary to apply
    """
    logger = logging.getLogger("preprocessing.config")
    logger.debug("🔄 Updating UI components dari config")
    
    try:
        # Extract config sections
        preprocessing_config = config.get('preprocessing', {})
        cleanup_config = config.get('cleanup', {})
        data_config = config.get('data', {})
        
        # Update preprocessing settings
        if resolution_dropdown := ui_components.get('resolution_dropdown'):
            if 'resolution' in preprocessing_config:
                resolution_dropdown.value = preprocessing_config['resolution']
            
        if normalization_dropdown := ui_components.get('normalization_dropdown'):
            if 'normalization' in preprocessing_config:
                normalization_dropdown.value = preprocessing_config['normalization']
            
        if preserve_aspect_checkbox := ui_components.get('preserve_aspect_checkbox'):
            if 'preserve_aspect' in preprocessing_config:
                preserve_aspect_checkbox.value = preprocessing_config['preserve_aspect']
            
        if target_splits_select := ui_components.get('target_splits_select'):
            if 'target_splits' in preprocessing_config:
                target_splits_select.value = preprocessing_config['target_splits']
            
        if batch_size_input := ui_components.get('batch_size_input'):
            if 'batch_size' in preprocessing_config:
                batch_size_input.value = str(preprocessing_config['batch_size'])
            
        if validation_checkbox := ui_components.get('validation_checkbox'):
            if 'validation' in preprocessing_config:
                validation_checkbox.value = preprocessing_config['validation']
            
        if move_invalid_checkbox := ui_components.get('move_invalid_checkbox'):
            if 'move_invalid' in preprocessing_config:
                move_invalid_checkbox.value = preprocessing_config['move_invalid']
            
        if invalid_dir_input := ui_components.get('invalid_dir_input'):
            if 'invalid_dir' in preprocessing_config:
                invalid_dir_input.value = preprocessing_config['invalid_dir']
            
        # Update cleanup settings
        if cleanup_target_dropdown := ui_components.get('cleanup_target_dropdown'):
            if 'target' in cleanup_config:
                cleanup_target_dropdown.value = cleanup_config['target']
            
        if backup_checkbox := ui_components.get('backup_checkbox'):
            if 'backup' in cleanup_config:
                backup_checkbox.value = cleanup_config['backup']
            
        # Update data settings
        if 'dir' in data_config:
            ui_components['data_dir'] = data_config['dir']
        
        logger.debug("✅ UI components updated dari config")
        
    except Exception as e:
        logger.error(f"❌ Error updating UI components: {str(e)}")
