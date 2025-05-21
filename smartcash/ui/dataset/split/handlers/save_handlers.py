"""
File: smartcash/ui/dataset/split/handlers/save_handlers.py
Deskripsi: Handler untuk menyimpan konfigurasi split dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.common.logger import get_logger
from smartcash.common.constants.log_messages import CONFIG_SAVED, CONFIG_ERROR
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.dataset.split.handlers.config_handlers import (
    load_config, save_config, update_ui_from_config, is_colab_environment
)
from smartcash.ui.dataset.split.handlers.ui_value_handlers import (
    get_ui_values, verify_config_consistency
)
from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel

logger = get_logger(__name__)

def get_default_base_dir():
    """Dapatkan direktori base default."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def handle_save_action(ui_components: Dict[str, Any]) -> None:
    """
    Handle aksi save konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status panel
        update_status_panel(ui_components, "Menyimpan konfigurasi split dataset...", 'info')
        
        # Dapatkan nilai dari UI
        ui_values = get_ui_values(ui_components)
        
        # Perbarui total jika perlu
        total = round(ui_values.get('train_ratio', 0) + ui_values.get('val_ratio', 0) + ui_values.get('test_ratio', 0), 2)
        if total != 1.0:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Total rasio tidak sama dengan 1.0: {total}")
            update_status_panel(ui_components, f"Peringatan: Total rasio tidak sama dengan 1.0: {total}", 'warning')
            
            # Normalisasi nilai rasio
            factor = 1.0 / total
            ui_values['train_ratio'] = round(ui_values['train_ratio'] * factor, 2)
            ui_values['val_ratio'] = round(ui_values['val_ratio'] * factor, 2)
            ui_values['test_ratio'] = round(ui_values['test_ratio'] * factor, 2)
            
            logger.info(f"{ICONS.get('info', 'ℹ️')} Normalisasi rasio: train={ui_values['train_ratio']}, val={ui_values['val_ratio']}, test={ui_values['test_ratio']}")
        
        # Dapatkan konfigurasi yang ada
        current_config = load_config()
        
        # Update konfigurasi
        if 'split' not in current_config:
            current_config['split'] = {}
        
        # Update konfigurasi dengan nilai UI
        current_config['split'].update(ui_values)
        
        # Simpan konfigurasi
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        save_success = config_manager.save_module_config('split', current_config)
        
        if not save_success:
            update_status_panel(ui_components, "Gagal menyimpan konfigurasi split dataset", 'error')
            logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi split dataset")
            return
        
        # Sinkronisasi dengan Google Drive jika di Colab
        drive_message = ""
        if is_colab_environment():
            try:
                # Cek apakah drive terpasang
                env_manager = get_environment_manager()
                if env_manager.is_drive_mounted:
                    # Sinkronisasi konfigurasi
                    success, message = config_manager.sync_to_drive('split')
                    if success:
                        drive_message = " dan disinkronkan dengan Google Drive"
                    else:
                        logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal sinkronisasi dengan Google Drive: {message}")
                        update_status_panel(ui_components, f"Konfigurasi disimpan, tetapi gagal sinkronisasi: {message}", 'warning')
                        return
                else:
                    logger.info(f"{ICONS.get('info', 'ℹ️')} Google Drive tidak terpasang, skip sinkronisasi")
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat sinkronisasi: {str(e)}")
        
        # Verifikasi konfigurasi tersimpan dengan benar
        saved_config = load_config()
        is_consistent, inconsistent_keys = verify_config_consistency(ui_values, saved_config)
        
        if is_consistent:
            update_status_panel(ui_components, f"Konfigurasi split dataset berhasil disimpan{drive_message}", 'success')
            logger.info(CONFIG_SAVED.format(config_name="split dataset"))
        else:
            update_status_panel(ui_components, f"Konfigurasi tersimpan tetapi tidak konsisten pada: {', '.join(inconsistent_keys)}", 'warning')
            logger.warning(f"{ICONS.get('warning', '⚠️')} Konfigurasi tidak konsisten pada: {', '.join(inconsistent_keys)}")
            
    except Exception as e:
        update_status_panel(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}", 'error')
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

# Import yang diperlukan untuk sinkronisasi
try:
    from smartcash.common.environment import get_environment_manager
except ImportError:
    # Fallback jika module tidak tersedia
    def get_environment_manager(*args, **kwargs):
        class DummyEnvironmentManager:
            def __init__(self):
                self.is_drive_mounted = False
        return DummyEnvironmentManager()
    logger.warning(f"{ICONS.get('warning', '⚠️')} smartcash.common.environment tidak tersedia, menggunakan dummy")