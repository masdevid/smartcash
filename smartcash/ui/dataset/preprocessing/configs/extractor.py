"""
File: smartcash/ui/dataset/preprocessing/configs/extractor.py
Deskripsi: Config extraction utilities untuk preprocessing module.
"""

from typing import Dict, Any, List
import logging


def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract preprocessing configuration dari UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Extracted configuration dictionary
    """
    logger = logging.getLogger("preprocessing.config")
    logger.debug("📋 Extracting preprocessing config dari UI components")
    
    config = {
        'preprocessing': {},
        'cleanup': {},
        'data': {}
    }
    
    try:
        # Extract preprocessing settings
        if resolution_dropdown := ui_components.get('resolution_dropdown'):
            config['preprocessing']['resolution'] = resolution_dropdown.value
            
        if normalization_dropdown := ui_components.get('normalization_dropdown'):
            config['preprocessing']['normalization'] = normalization_dropdown.value
            
        if preserve_aspect_checkbox := ui_components.get('preserve_aspect_checkbox'):
            config['preprocessing']['preserve_aspect'] = preserve_aspect_checkbox.value
            
        if target_splits_select := ui_components.get('target_splits_select'):
            config['preprocessing']['target_splits'] = target_splits_select.value
            
        if batch_size_input := ui_components.get('batch_size_input'):
            config['preprocessing']['batch_size'] = int(batch_size_input.value)
            
        if validation_checkbox := ui_components.get('validation_checkbox'):
            config['preprocessing']['validation'] = validation_checkbox.value
            
        if move_invalid_checkbox := ui_components.get('move_invalid_checkbox'):
            config['preprocessing']['move_invalid'] = move_invalid_checkbox.value
            
        if invalid_dir_input := ui_components.get('invalid_dir_input'):
            config['preprocessing']['invalid_dir'] = invalid_dir_input.value
            
        # Extract cleanup settings
        if cleanup_target_dropdown := ui_components.get('cleanup_target_dropdown'):
            config['cleanup']['target'] = cleanup_target_dropdown.value
            
        if backup_checkbox := ui_components.get('backup_checkbox'):
            config['cleanup']['backup'] = backup_checkbox.value
            
        # Extract data settings
        data_dir = ui_components.get('data_dir', 'data')
        config['data']['dir'] = data_dir
        
        logger.debug(f"📋 Extracted config with {len(config)} sections")
        
    except Exception as e:
        logger.error(f"❌ Error extracting config: {str(e)}")
        
    return config
