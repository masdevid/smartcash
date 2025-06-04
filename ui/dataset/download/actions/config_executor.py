"""
File: smartcash/ui/dataset/download/actions/config_executor.py
Deskripsi: Config action executors untuk save dan reset dengan ConfigHandler integration
"""

from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager

def execute_save_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute save config action."""
    logger = ui_components.get('logger')
    config_handler = ui_components.get('config_handler')
    
    if button: button.disabled = True
    
    try:
        if config_handler:
            # Use ConfigHandler
            success = config_handler.save_config(ui_components)
            if success:
                logger and logger.success("✅ Konfigurasi berhasil disimpan")
            else:
                logger and logger.error("❌ Gagal menyimpan konfigurasi")
        else:
            # Fallback method
            config = _extract_config_from_ui(ui_components)
            success = _save_config_safe(config, logger)
            
            if success:
                logger and logger.success("✅ Konfigurasi berhasil disimpan")
            else:
                logger and logger.error("❌ Gagal menyimpan konfigurasi")
                
    except Exception as e:
        logger and logger.error(f"❌ Error save config: {str(e)}")
    finally:
        if button: button.disabled = False

def execute_reset_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute reset config action."""
    logger = ui_components.get('logger')
    config_handler = ui_components.get('config_handler')
    
    if button: button.disabled = True
    
    try:
        if config_handler:
            # Use ConfigHandler
            success = config_handler.reset_config(ui_components)
            if success:
                logger and logger.success("✅ Konfigurasi berhasil direset")
            else:
                logger and logger.error("❌ Gagal reset konfigurasi")
        else:
            # Fallback method
            _reset_to_defaults(ui_components)
            logger and logger.success("✅ Form berhasil direset ke nilai default")
            
    except Exception as e:
        logger and logger.error(f"❌ Error reset config: {str(e)}")
    finally:
        if button: button.disabled = False

def _extract_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components."""
    config = {}
    
    field_mapping = {
        'workspace': 'workspace', 'project': 'project', 'version': 'version',
        'output_dir': 'output_dir', 'backup_dir': 'backup_dir',
        'backup_checkbox': 'backup_before_download', 'organize_dataset': 'organize_dataset'
    }
    
    for ui_key, config_key in field_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = ui_components[ui_key].value
            if value and (isinstance(value, bool) or (isinstance(value, str) and value.strip())):
                config[config_key] = value.strip() if isinstance(value, str) else value
    
    return config

def _save_config_safe(config: Dict[str, Any], logger=None) -> bool:
    """Save config dengan error handling."""
    try:
        config_manager = get_config_manager()
        
        if hasattr(config_manager, 'save_config'):
            return config_manager.save_config(config, 'download')
        elif hasattr(config_manager, 'config'):
            if not hasattr(config_manager, 'config') or config_manager.config is None:
                config_manager.config = {}
            config_manager.config.update({'download': config})
            return True
        else:
            logger and logger.warning("⚠️ Config manager tidak memiliki method save")
            return False
            
    except Exception as e:
        logger and logger.error(f"❌ Error save config: {str(e)}")
        return False

def _reset_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default values."""
    # Get defaults dari stored defaults atau create new
    defaults = ui_components.get('_config_defaults', _get_fallback_defaults())
    
    field_mapping = {
        'workspace': 'workspace', 'project': 'project', 'version': 'version',
        'output_dir': 'output_dir', 'backup_dir': 'backup_dir',
        'backup_before_download': 'backup_checkbox', 'organize_dataset': 'organize_dataset'
    }
    
    for config_key, ui_key in field_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            default_value = defaults.get(config_key, '')
            ui_components[ui_key].value = default_value
    
    # Reset API key
    if 'api_key' in ui_components and hasattr(ui_components['api_key'], 'value'):
        ui_components['api_key'].value = _detect_api_key()

def _get_fallback_defaults() -> Dict[str, Any]:
    """Get fallback defaults jika tidak ada stored defaults."""
    api_key = _detect_api_key()
    
    return {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'api_key': api_key,
        'output_dir': 'data/downloads',
        'backup_dir': 'data/backup',
        'backup_before_download': False,
        'organize_dataset': True
    }

def _detect_api_key() -> str:
    """Detect API key dari environment."""
    import os
    
    # Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key: return api_key
    
    # Google Colab userdata
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY', '')
    except: return ''