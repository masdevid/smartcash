"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_service_handler.py
Deskripsi: Handler untuk preprocessing service
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.handlers.config_handler import get_preprocessing_config
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def handle_preprocessing_service(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle preprocessing service.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Get config
        config = get_preprocessing_config(ui_components)
        
        # Update UI components
        if 'preprocess_options' in ui_components:
            preproc_options = ui_components['preprocess_options']
            if hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
                # Update image size
                preproc_options.children[0].value = config['preprocessing']['img_size']
                
                # Update normalization options
                preproc_options.children[1].value = config['preprocessing']['normalization']['enabled']
                preproc_options.children[2].value = config['preprocessing']['normalization']['preserve_aspect_ratio']
                
                # Update cache dan workers
                preproc_options.children[3].value = config['preprocessing']['enabled']
                preproc_options.children[4].value = config['preprocessing']['num_workers']
        
        # Update validation options
        if 'validation_options' in ui_components:
            validation_options = ui_components['validation_options']
            if hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
                validation_options.children[0].value = config['preprocessing']['validate']['enabled']
                validation_options.children[1].value = config['preprocessing']['validate']['fix_issues']
                validation_options.children[2].value = config['preprocessing']['validate']['move_invalid']
                validation_options.children[3].value = config['preprocessing']['validate']['invalid_dir']
        
        # Update split selector
        if 'split_selector' in ui_components:
            split_selector = ui_components['split_selector']
            if hasattr(split_selector, 'value'):
                split_map = {
                    'All Splits': ['train', 'valid', 'test'],
                    'Train Only': ['train'],
                    'Validation Only': ['valid'],
                    'Test Only': ['test']
                }
                for key, value in split_map.items():
                    if value == config['preprocessing']['splits']:
                        split_selector.value = key
                        break
        
        logger.info("✅ Preprocessing service berhasil diupdate")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat update preprocessing service: {str(e)}")
        return ui_components
