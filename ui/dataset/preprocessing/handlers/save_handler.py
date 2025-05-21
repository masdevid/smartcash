"""
File: smartcash/ui/dataset/preprocessing/handlers/save_handler.py
Deskripsi: Handler untuk tombol save konfigurasi preprocessing dataset
"""

from typing import Dict, Any, Optional

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handler import get_preprocessing_config_from_ui

def handle_save_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol save konfigurasi preprocessing.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Log save konfigurasi
        log_message(ui_components, "Menyimpan konfigurasi preprocessing dataset...", "info", "💾")
        
        # Update UI state
        update_status_panel(ui_components, "info", "Menyimpan konfigurasi preprocessing...")
        
        # Save konfigurasi
        save_preprocessing_config(ui_components)
        
        # Log hasil
        log_message(ui_components, "Konfigurasi preprocessing berhasil disimpan", "success", "✅")
        
        # Update UI state
        update_status_panel(ui_components, "success", "Konfigurasi berhasil disimpan")
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_status_panel(ui_components, "error", f"Error saat simpan konfigurasi: {error_message}")
        log_message(ui_components, f"Error saat simpan konfigurasi preprocessing: {error_message}", "error", "❌")
    
    finally:
        # Re-enable tombol setelah operasi selesai
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def save_preprocessing_config(ui_components: Dict[str, Any]) -> None:
    """
    Simpan konfigurasi preprocessing ke config manager.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Import config manager
        from smartcash.common.config import get_config_manager
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Get konfigurasi preprocessing dari UI
        preprocessing_config = get_preprocessing_config_from_ui(ui_components)
        
        # Get module config
        dataset_config = config_manager.get_module_config('dataset')
        
        # Update config preprocessing
        if 'preprocessing' not in dataset_config:
            dataset_config['preprocessing'] = {}
        
        # Update preprocessing config
        dataset_config['preprocessing'].update(preprocessing_config)
        
        # Save config
        config_manager.save_module_config('dataset', dataset_config)
        
        # Log proses yang dilakukan
        log_message(ui_components, "Konfigurasi preprocessing berhasil disimpan ke config manager", "info", "💾")
        
        # Sync ke drive jika tersedia
        if hasattr(config_manager, 'sync_to_drive') and callable(config_manager.sync_to_drive):
            log_message(ui_components, "Sinkronisasi konfigurasi ke Google Drive...", "info", "🔄")
            config_manager.sync_to_drive()
            log_message(ui_components, "Konfigurasi berhasil disinkronkan ke Google Drive", "success", "✅")
            
    except Exception as e:
        # Re-raise exception untuk ditangani oleh handler
        raise Exception(f"Gagal menyimpan konfigurasi: {str(e)}") 