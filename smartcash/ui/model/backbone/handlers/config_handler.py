"""
File: smartcash/ui/model/backbone/handlers/config_handler.py
Deskripsi: Handler untuk konfigurasi backbone model
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def get_backbone_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get konfigurasi backbone model.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi backbone model
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_module_config('model')
        
        # Ensure config structure
        if not config:
            config = get_default_backbone_config()
        elif 'backbone' not in config:
            config['backbone'] = get_default_backbone_config()['backbone']
            
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat get backbone config: {str(e)}")
        return get_default_backbone_config()

def get_default_backbone_config() -> Dict[str, Any]:
    """
    Get konfigurasi default backbone model.
    
    Returns:
        Dictionary konfigurasi default backbone model
    """
    return {
        'backbone': {
            'name': 'yolov8n',
            'pretrained': True,
            'freeze_backbone': False,
            'freeze_layers': [],
            'input_size': 640,
            'channels': 3,
            'num_classes': 80,
            'weights': None
        }
    }

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get current config
        config = get_backbone_config(ui_components)
        
        # Update backbone options
        if 'backbone_options' in ui_components:
            backbone_options = ui_components['backbone_options']
            if hasattr(backbone_options, 'children') and len(backbone_options.children) >= 4:
                # Update model name
                config['backbone']['name'] = backbone_options.children[0].value
                
                # Update pretrained checkbox
                config['backbone']['pretrained'] = backbone_options.children[1].value
                
                # Update freeze backbone checkbox
                config['backbone']['freeze_backbone'] = backbone_options.children[2].value
                
                # Update input size
                config['backbone']['input_size'] = backbone_options.children[3].value
        
        # Update advanced options
        if 'advanced_options' in ui_components:
            advanced_options = ui_components['advanced_options']
            if hasattr(advanced_options, 'children') and len(advanced_options.children) >= 3:
                # Update channels
                config['backbone']['channels'] = advanced_options.children[0].value
                
                # Update num classes
                config['backbone']['num_classes'] = advanced_options.children[1].value
                
                # Update weights path
                config['backbone']['weights'] = advanced_options.children[2].value
            
        # Save config
        config_manager = get_config_manager()
        config_manager.set_module_config('model', config)
        
        logger.info("✅ Konfigurasi backbone berhasil diupdate dari UI")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update config dari UI: {str(e)}")
        return get_backbone_config(ui_components)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Ensure config structure
        if not config:
            config = get_default_backbone_config()
        elif 'backbone' not in config:
            config['backbone'] = get_default_backbone_config()['backbone']
            
        # Update UI components
        if 'backbone_options' in ui_components:
            backbone_options = ui_components['backbone_options']
            if hasattr(backbone_options, 'children') and len(backbone_options.children) >= 4:
                # Update model name
                backbone_options.children[0].value = config['backbone']['name']
                
                # Update pretrained checkbox
                backbone_options.children[1].value = config['backbone']['pretrained']
                
                # Update freeze backbone checkbox
                backbone_options.children[2].value = config['backbone']['freeze_backbone']
                
                # Update input size
                backbone_options.children[3].value = config['backbone']['input_size']
        
        # Update advanced options
        if 'advanced_options' in ui_components:
            advanced_options = ui_components['advanced_options']
            if hasattr(advanced_options, 'children') and len(advanced_options.children) >= 3:
                # Update channels
                advanced_options.children[0].value = config['backbone']['channels']
                
                # Update num classes
                advanced_options.children[1].value = config['backbone']['num_classes']
                
                # Update weights path
                advanced_options.children[2].value = config['backbone']['weights']
            
        logger.info("✅ UI backbone berhasil diupdate dari konfigurasi")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat update UI dari config: {str(e)}")
        return ui_components 