# File: smartcash/ui/pretrained/handlers/config_updater.py
"""
File: smartcash/ui/pretrained/handlers/config_updater.py
Deskripsi: Config updater untuk pretrained models dengan safe update approach
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def update_pretrained_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """🔄 Update UI components dengan konfigurasi yang dimuat
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi untuk diterapkan
        
    Raises:
        ValueError: Jika update UI gagal atau data tidak valid
    """
    if not ui_components or not isinstance(ui_components, dict):
        raise ValueError("❌ UI components tidak valid untuk update")
    
    if not config or not isinstance(config, dict):
        raise ValueError("❌ Config tidak valid untuk update UI")
    
    try:
        pretrained_config = config.get('pretrained_models', {})
        
        # Update models directory
        if 'models_dir_input' in ui_components and 'models_dir' in pretrained_config:
            ui_components['models_dir_input'].value = pretrained_config['models_dir']
            logger.debug(f"🔧 Updated models_dir: {pretrained_config['models_dir']}")
        
        # Update drive models directory
        if 'drive_models_dir_input' in ui_components and 'drive_models_dir' in pretrained_config:
            ui_components['drive_models_dir_input'].value = pretrained_config['drive_models_dir']
            logger.debug(f"🔧 Updated drive_models_dir: {pretrained_config['drive_models_dir']}")
        
        # Update pretrained type
        if 'pretrained_type_dropdown' in ui_components and 'pretrained_type' in pretrained_config:
            ui_components['pretrained_type_dropdown'].value = pretrained_config['pretrained_type']
            logger.debug(f"🔧 Updated pretrained_type: {pretrained_config['pretrained_type']}")
        
        # Update sync drive checkbox
        if 'sync_drive_checkbox' in ui_components and 'sync_drive' in pretrained_config:
            ui_components['sync_drive_checkbox'].value = pretrained_config['sync_drive']
            logger.debug(f"🔧 Updated sync_drive: {pretrained_config['sync_drive']}")
        
        logger.info("✅ UI berhasil diupdate dengan konfigurasi yang dimuat")
        
    except Exception as e:
        error_msg = f"❌ Gagal mengupdate UI dengan config: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def reset_pretrained_ui(ui_components: Dict[str, Any]) -> None:
    """🔄 Reset UI components ke nilai default
    
    Args:
        ui_components: Dictionary berisi komponen UI untuk direset
        
    Raises:
        ValueError: Jika reset UI gagal
    """
    if not ui_components or not isinstance(ui_components, dict):
        raise ValueError("❌ UI components tidak valid untuk reset")
    
    try:
        from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
        
        # Get default config
        default_config = get_default_pretrained_config()
        
        # Apply default config to UI
        update_pretrained_ui(ui_components, default_config)
        
        logger.info("✅ UI berhasil direset ke nilai default")
        
    except Exception as e:
        error_msg = f"❌ Gagal mereset UI: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def apply_config_to_ui(ui_components: Dict[str, Any], config_key: str, config_value: Any) -> bool:
    """🎯 Apply single config value ke UI component yang sesuai
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config_key: Key konfigurasi yang akan diterapkan
        config_value: Value konfigurasi yang akan diterapkan
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Mapping config key ke UI component
        config_mapping = {
            'models_dir': 'models_dir_input',
            'sync_drive': 'sync_drive_checkbox',
            'model_urls': {
                'yolov5s': 'yolo_url_input',
                'efficientnet': 'efficientnet_url_input'
            }
        }
        
        # Handle model URLs specially
        if config_key == 'model_urls' and isinstance(config_value, dict):
            url_mapping = config_mapping.get('model_urls', {})
            updated = False
            for model_name, url in config_value.items():
                ui_key = url_mapping.get(model_name)
                if ui_key and ui_key in ui_components:
                    ui_components[ui_key].value = url
                    logger.debug(f"🔧 Applied {model_name} URL: {url}")
                    updated = True
            return updated
            
        # Handle regular config values
        ui_key = config_mapping.get(config_key)
        if not ui_key or ui_key not in ui_components:
            logger.warning(f" UI component tidak ditemukan untuk key: {config_key}")
            return False
        
        # Apply value ke UI component
        ui_components[ui_key].value = config_value
        logger.debug(f"🔧 Applied {config_key}: {config_value}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error applying config {config_key}: {str(e)}")
        return False