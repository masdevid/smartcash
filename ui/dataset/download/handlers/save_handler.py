"""
File: smartcash/ui/dataset/download/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi download dataset ke SimpleConfigManager
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger

def handle_save_config(ui_components: Dict[str, Any], button=None) -> None:
    """
    Handler untuk menyimpan konfigurasi download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Tombol yang diklik (opsional)
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Nonaktifkan tombol selama proses
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Log info
        log_message(ui_components, "Menyimpan konfigurasi dataset...", "info", "💾")
        
        # Ambil nilai dari komponen UI
        config = _get_config_from_ui(ui_components)
        
        # Simpan ke SimpleConfigManager
        config_manager = get_config_manager()
        
        # Simpan ke modul dataset
        result = config_manager.save_module_config('dataset', config)
        
        if result:
            log_message(ui_components, "Konfigurasi dataset berhasil disimpan", "success", "✅")
            
            # Update status panel jika tersedia
            if 'status_panel' in ui_components:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(
                    ui_components['status_panel'],
                    "Konfigurasi dataset berhasil disimpan",
                    "success"
                )
        else:
            log_message(ui_components, "Gagal menyimpan konfigurasi dataset", "error", "❌")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}", "error", "❌")
    finally:
        # Aktifkan kembali tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def _get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ambil konfigurasi dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict[str, Any]: Konfigurasi yang diambil dari UI
    """
    config = {}
    
    # Daftar kunci yang akan diambil dari UI
    config_keys = [
        'workspace', 'project', 'version', 'api_key', 
        'output_dir', 'backup_dir'
    ]
    
    # Checkbox dan status
    checkbox_keys = [
        'validate_dataset', 'backup_checkbox', 'save_logs'
    ]
    
    # Ekstrak nilai dari UI components
    for key in config_keys:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            # Jangan simpan API key jika kosong atau dimulai dengan '*'
            if key == 'api_key' and (not ui_components[key].value or ui_components[key].value.startswith('*')):
                continue
            
            config[key] = ui_components[key].value
    
    # Ekstrak checkbox
    for key in checkbox_keys:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            config[key] = ui_components[key].value
    
    return config 