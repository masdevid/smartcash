"""
File: smartcash/ui/dataset/download/handlers/save_action.py
Deskripsi: Fixed save action yang mengatasi error 'dict' object has no attribute 'endswith'
"""

from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager

def execute_save_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi save konfigurasi dengan error handling yang diperbaiki."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("💾 Menyimpan konfigurasi")
    
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Ambil config dari UI
        config = _extract_config_from_ui(ui_components)
        
        # Simpan via config manager dengan error handling
        success = _save_config_safe(config, logger)
        
        if success and logger:
            logger.success("✅ Konfigurasi berhasil disimpan")
        elif logger:
            logger.error("❌ Gagal menyimpan konfigurasi")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error save config: {str(e)}")
    finally:
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def _save_config_safe(config: Dict[str, Any], logger=None) -> bool:
    """Save config dengan error handling yang aman."""
    try:
        config_manager = get_config_manager()
        
        # Method 1: Gunakan save_config jika tersedia
        if hasattr(config_manager, 'save_config'):
            return config_manager.save_config(config, 'dataset')  # Fix parameter order
        
        # Method 2: Gunakan method lama dengan dict merge
        elif hasattr(config_manager, 'config'):
            if not hasattr(config_manager, 'config') or config_manager.config is None:
                config_manager.config = {}
            config_manager.config.update({'dataset': config})
            return True
        
        # Method 3: Direct save menggunakan config_cache
        elif hasattr(config_manager, 'config_cache'):
            config_manager.config_cache['dataset'] = config
            return True
        
        else:
            if logger:
                logger.warning("⚠️ Config manager tidak memiliki method save yang dikenali")
            return False
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error dalam save_config_safe: {str(e)}")
        return False

def _extract_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstrak konfigurasi dari UI components dengan validasi."""
    config = {}
    
    # Field mapping yang aman
    field_map = {
        'workspace': 'workspace',
        'project': 'project', 
        'version': 'version',
        'output_dir': 'output_dir',
        'backup_dir': 'backup_dir'
    }
    
    checkbox_map = {
        'validate_dataset': 'validate_dataset',
        'backup_checkbox': 'backup_before_download'
    }
    
    # Extract text fields dengan validasi
    for ui_key, config_key in field_map.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = ui_components[ui_key].value
            if value and isinstance(value, str):  # Hanya save string yang valid
                config[config_key] = value.strip()  # Strip whitespace
    
    # Extract checkboxes dengan validasi
    for ui_key, config_key in checkbox_map.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = ui_components[ui_key].value
            if isinstance(value, bool):  # Hanya save boolean yang valid
                config[config_key] = value
    
    # API key - handle secara khusus (sensitive data)
    if 'api_key' in ui_components and hasattr(ui_components['api_key'], 'value'):
        api_key = ui_components['api_key'].value
        if api_key and isinstance(api_key, str) and len(api_key.strip()) > 0:
            # Tidak save API key ke config file untuk keamanan
            # config['api_key'] = api_key.strip()
            pass
    
    return config