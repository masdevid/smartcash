"""
File: smartcash/ui/dataset/augmentation/handlers/setup_handlers.py
Deskripsi: Setup handler untuk augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.handlers.config_handler import get_augmentation_config
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment (opsional)
        config: Konfigurasi (opsional)
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Get config
        if config is None:
            config = get_augmentation_config(ui_components)
        
        # Update UI components
        if 'augmentation_options' in ui_components:
            aug_options = ui_components['augmentation_options']
            if hasattr(aug_options, 'children') and len(aug_options.children) >= 4:
                # Update enabled checkbox
                aug_options.children[0].value = config['augmentation']['enabled']
                
                # Update num variations slider
                aug_options.children[1].value = config['augmentation']['num_variations']
                
                # Update output prefix
                aug_options.children[2].value = config['augmentation']['output_prefix']
                
                # Update process bboxes checkbox
                aug_options.children[3].value = config['augmentation']['process_bboxes']
        
        # Update output options
        if 'output_options' in ui_components:
            output_options = ui_components['output_options']
            if hasattr(output_options, 'children') and len(output_options.children) >= 4:
                # Update output dir
                output_options.children[0].value = config['augmentation']['output_dir']
                
                # Update validate checkbox
                output_options.children[1].value = config['augmentation']['validate_results']
                
                # Update resume checkbox
                output_options.children[2].value = config['augmentation']['resume']
                
                # Update num workers slider
                output_options.children[3].value = config['augmentation']['num_workers']
        
        # Update balance options
        if 'balance_options' in ui_components:
            balance_options = ui_components['balance_options']
            if hasattr(balance_options, 'children') and len(balance_options.children) >= 2:
                # Update balance classes checkbox
                balance_options.children[0].value = config['augmentation']['balance_classes']
                
                # Update target count slider
                balance_options.children[1].value = config['augmentation']['target_count']
        
        logger.info("✅ Augmentation handlers berhasil disetup")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat setup augmentation handlers: {str(e)}")
        return ui_components

def setup_state_handler(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handler untuk state augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state
    
    # Deteksi state augmentasi
    ui_components = detect_augmentation_state(ui_components)
    
    return ui_components
