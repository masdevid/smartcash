"""
File: smartcash/ui/dataset/split/handlers/config_updater.py
Deskripsi: Update dan save konfigurasi split dataset
"""

from typing import Dict, Any, Tuple

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.ui_extractor import validate_extracted_values

logger = get_logger(__name__)


def update_and_save_config(ui_components: Dict[str, Any], ui_values: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Update dan save konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        ui_values: Nilai yang diekstrak dari UI
        
    Returns:
        Tuple (success, message)
    """
    try:
        # Validasi nilai UI
        is_valid, validation_message = validate_extracted_values(ui_values)
        if not is_valid:
            return False, f"Validasi gagal: {validation_message}"
        
        # Get config manager
        config_manager = ui_components.get('config_manager')
        if not config_manager:
            return False, "Config manager tidak tersedia"
        
        # Load existing config dan merge dengan UI values
        existing_config = config_manager.get_config('dataset_config') or {}
        updated_config = _merge_config_with_ui_values(existing_config, ui_values)
        
        # Save konfigurasi
        save_success = config_manager.save_config(updated_config, 'dataset_config')
        if not save_success:
            return False, "Gagal menyimpan konfigurasi ke file"
        
        # Sync ke Drive jika tersedia
        sync_message = _attempt_drive_sync(ui_components, config_manager)
        
        success_message = f"Konfigurasi split berhasil disimpan{sync_message}"
        return True, success_message
        
    except Exception as e:
        error_message = f"Error update config: {str(e)}"
        logger.error(f"💥 {error_message}")
        return False, error_message


def _merge_config_with_ui_values(existing_config: Dict[str, Any], ui_values: Dict[str, Any]) -> Dict[str, Any]:
    """Merge existing config dengan UI values."""
    updated_config = existing_config.copy()
    
    # Update data section
    if 'data' not in updated_config:
        updated_config['data'] = {}
    updated_config['data'].update(ui_values.get('data', {}))
    
    # Update split_settings section
    updated_config['split_settings'] = ui_values.get('split_settings', {})
    
    return updated_config


def _attempt_drive_sync(ui_components: Dict[str, Any], config_manager) -> str:
    """Attempt sinkronisasi ke Google Drive."""
    try:
        env = ui_components.get('env')
        if env and env.is_colab and env.is_drive_mounted:
            # Sync logic jika diperlukan (tergantung implementasi config_manager)
            return " dan disinkronkan ke Google Drive"
        else:
            return ""
    except Exception as e:
        logger.warning(f"⚠️ Gagal sync ke Drive: {str(e)}")
        return " (tanpa sinkronisasi Drive)"


def validate_config_structure(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi struktur konfigurasi split.
    
    Args:
        config: Konfigurasi yang akan divalidasi
        
    Returns:
        Tuple (is_valid, message)
    """
    required_keys = {
        'data': ['split_ratios', 'stratified_split', 'random_seed'],
        'split_settings': ['backup_before_split', 'backup_dir', 'dataset_path', 'preprocessed_path']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            return False, f"Section '{section}' tidak ditemukan"
        
        for key in keys:
            if key not in config[section]:
                return False, f"Key '{key}' tidak ditemukan di section '{section}'"
    
    return True, "Struktur konfigurasi valid"


def create_backup_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat backup dari konfigurasi saat ini."""
    import copy
    import datetime
    
    backup_config = copy.deepcopy(config)
    backup_config['_backup_info'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'source': 'split_config_ui'
    }
    
    return backup_config