"""
File: smartcash/ui/dataset/download/handlers/reset_handler.py
Deskripsi: Handler untuk mereset UI download dataset ke nilai default
"""

from typing import Dict, Any, Optional
from IPython.display import display
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger

def handle_reset_button_click(ui_components: Dict[str, Any], button=None) -> None:
    """
    Handler untuk mereset UI download dataset ke nilai default.
    
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
        log_message(ui_components, "Mereset konfigurasi dataset...", "info", "🔄")
        
        # Reset UI ke nilai default
        _reset_ui_to_defaults(ui_components)
        
        # Update status panel jika tersedia
        if 'status_panel' in ui_components:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(
                ui_components['status_panel'],
                "Konfigurasi dataset berhasil direset",
                "success"
            )
        
        # Log sukses
        log_message(ui_components, "Konfigurasi dataset berhasil direset", "success", "✅")
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat mereset konfigurasi: {str(e)}", "error", "❌")
    finally:
        # Aktifkan kembali tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def _reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI ke nilai default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Ambil default values dari SimpleConfigManager jika tersedia
    try:
        config_manager = get_config_manager()
        default_config = config_manager.get_module_config('dataset_defaults')
        
        if default_config:
            # Reset UI dengan default config
            _update_ui_from_config(ui_components, default_config)
            return
    except Exception as e:
        log_message(ui_components, f"Tidak bisa memuat konfigurasi default: {str(e)}", "debug", "ℹ️")
    
    # Fallback ke defaults hardcoded jika tidak ada config
    _set_hardcoded_defaults(ui_components)

def _update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan diterapkan
    """
    # Update widget UI berdasarkan config yang tersedia
    for key, value in config.items():
        # Cari komponen UI yang sesuai dengan key
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception as e:
                log_message(ui_components, f"Tidak bisa mengatur nilai '{key}': {str(e)}", "debug", "⚠️")

def _set_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """
    Set nilai default hardcoded ke UI components.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Default Roboflow
    if 'workspace' in ui_components:
        ui_components['workspace'].value = ""
    
    if 'project' in ui_components:
        ui_components['project'].value = ""
    
    if 'version' in ui_components:
        ui_components['version'].value = ""
    
    # Masking API key jika ada
    if 'api_key' in ui_components:
        current_key = ui_components['api_key'].value
        if current_key and len(current_key) > 4 and not current_key.startswith('*'):
            # Mask existing API key
            ui_components['api_key'].value = f"****{current_key[-4:]}"
        else:
            # Clear jika itu sudah masked
            ui_components['api_key'].value = ""
    
    # Default output path
    if 'output_dir' in ui_components:
        ui_components['output_dir'].value = "data"
    
    # Default backup path
    if 'backup_dir' in ui_components:
        ui_components['backup_dir'].value = "data_backup"
    
    # Default checkbox
    if 'validate_dataset' in ui_components:
        ui_components['validate_dataset'].value = True
    
    if 'backup_checkbox' in ui_components:
        ui_components['backup_checkbox'].value = True
