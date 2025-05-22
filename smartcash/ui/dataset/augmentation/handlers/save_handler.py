"""
File: smartcash/ui/dataset/augmentation/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi augmentasi (tanpa move_to_preprocessed)
"""

from typing import Dict, Any
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel

def handle_save_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol save konfigurasi augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Nonaktifkan tombol selama proses
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        log_message(ui_components, "Menyimpan konfigurasi augmentasi...", "info", "💾")
        
        # Ambil konfigurasi dari UI
        config = _get_config_from_ui(ui_components)
        
        # Simpan menggunakan ConfigManager
        config_manager = get_config_manager()
        result = config_manager.save_module_config('augmentation', config)
        
        if result:
            log_message(ui_components, "Konfigurasi augmentasi berhasil disimpan", "success", "✅")
            update_status_panel(ui_components, "Konfigurasi berhasil disimpan", "success")
        else:
            log_message(ui_components, "Gagal menyimpan konfigurasi", "error", "❌")
            update_status_panel(ui_components, "Gagal menyimpan konfigurasi", "error")
            
    except Exception as e:
        log_message(ui_components, f"Error saat menyimpan: {str(e)}", "error", "❌")
        update_status_panel(ui_components, f"Error saat menyimpan: {str(e)}", "error")
    finally:
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def _get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ambil konfigurasi dari komponen UI (tanpa move_to_preprocessed)."""
    config = {}
    
    config_keys = [
        'num_variations', 'target_count', 'output_prefix', 
        'balance_classes', 'validate_results'
    ]
    
    for key in config_keys:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            config[key] = ui_components[key].value
    
    # Jenis augmentasi
    if 'augmentation_types' in ui_components and hasattr(ui_components['augmentation_types'], 'value'):
        config['types'] = list(ui_components['augmentation_types'].value)
    
    return config