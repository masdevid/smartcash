"""
File: smartcash/ui/dataset/split/handlers/save_handlers.py
Deskripsi: Handler untuk save konfigurasi di split dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.common.constants.log_messages import (
    CONFIG_SAVED, CONFIG_ERROR, OPERATION_SUCCESS, OPERATION_FAILED
)
from smartcash.ui.dataset.split.handlers.config_handlers import load_config, save_config, is_colab_environment
from smartcash.ui.dataset.split.handlers.ui_value_handlers import (
    get_ui_values, create_config_from_ui_values, verify_config_consistency, merge_config_with_ui_values
)
from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel

logger = get_logger(__name__)

def handle_save_action(ui_components: Dict[str, Any]) -> None:
    """
    Handle aksi save konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Menyimpan konfigurasi...", 'info')
        
        # Dapatkan nilai dari UI
        ui_values = get_ui_values(ui_components)
        
        # Dapatkan konfigurasi yang ada
        current_config = load_config()
        
        # Gabungkan konfigurasi yang ada dengan nilai UI
        config = merge_config_with_ui_values(current_config, ui_values)
        
        # Simpan konfigurasi
        saved_config = save_config(config, ui_components)
        
        # Verifikasi konfigurasi tersimpan dengan benar
        if saved_config and 'split' in saved_config:
            # Muat ulang konfigurasi untuk verifikasi
            loaded_config = load_config()
            
            # Periksa apakah data yang disimpan sesuai
            is_consistent, inconsistent_keys = verify_config_consistency(ui_values, loaded_config)
            
            if is_consistent:
                # Pesan sukses yang berbeda berdasarkan lingkungan
                is_colab = is_colab_environment()
                if is_colab:
                    success_message = "Konfigurasi berhasil disimpan dan disinkronkan dengan Google Drive"
                else:
                    success_message = "Konfigurasi berhasil disimpan"
                    
                # Update status
                update_status_panel(ui_components, success_message, 'success')
                logger.info(OPERATION_SUCCESS.format(operation="Penyimpanan konfigurasi"))
            else:
                # Update status
                warning_message = f"Inkonsistensi data pada: {', '.join(inconsistent_keys)}"
                update_status_panel(ui_components, warning_message, 'warning')
                logger.warning(f"⚠️ {warning_message}")
        else:
            # Update status
            error_message = "Gagal menyimpan konfigurasi"
            update_status_panel(ui_components, error_message, 'error')
            logger.error(OPERATION_FAILED.format(operation="Penyimpanan konfigurasi", reason="Format konfigurasi tidak valid"))
    except Exception as e:
        # Update status
        error_message = f"Error saat menyimpan: {str(e)}"
        update_status_panel(ui_components, error_message, 'error')
        logger.error(CONFIG_ERROR.format(operation="menyimpan", error=str(e)))

def create_save_handler(ui_components: Dict[str, Any]):
    """
    Buat handler untuk tombol save.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Function handler untuk tombol save
    """
    def on_save_clicked(b):
        handle_save_action(ui_components)
    
    return on_save_clicked 