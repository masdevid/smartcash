"""
File: smartcash/ui/dataset/download/handlers/save_action.py
Deskripsi: Handler aksi save konfigurasi download
"""

from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager

def execute_save_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi save konfigurasi."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("💾 Menyimpan konfigurasi")
    
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Ambil config dari UI
        config = _extract_config_from_ui(ui_components)
        
        # Simpan via config manager
        config_manager = get_config_manager()
        if hasattr(config_manager, 'save_config'):
            success = config_manager.save_config('dataset', config)
        else:
            # Fallback untuk config manager lama
            config_manager.config.update({'dataset': config})
            success = True
        
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

def _extract_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstrak konfigurasi dari UI components."""
    config = {}
    
    # Field mapping
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
    
    # Extract text fields
    for ui_key, config_key in field_map.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = ui_components[ui_key].value
            if value:  # Only save non-empty values
                config[config_key] = value
    
    # Extract checkboxes
    for ui_key, config_key in checkbox_map.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            config[config_key] = ui_components[ui_key].value
    
    return config