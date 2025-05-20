"""
File: smartcash/ui/training_config/training_strategy/handlers/button_handlers.py
Deskripsi: Handler untuk tombol pada komponen UI strategi pelatihan
"""

from typing import Dict, Any, Optional, Callable
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.training_strategy.handlers.drive_handlers import sync_to_drive, sync_from_drive
from smartcash.ui.training_config.training_strategy.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    get_default_config,
    update_training_strategy_info
)

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def setup_training_strategy_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI strategi pelatihan.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager(base_dir=get_default_base_dir())
        
        # Dapatkan config manager
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Validasi config
        if config is None:
            config = config_manager.get_module_config('training_strategy', {})
        
        # Pastikan ui_components memiliki referensi ke config
        ui_components['config'] = config
        
        # Handler untuk tombol save
        def on_save_click(b):
            status_panel = ui_components.get('status_panel', ui_components.get('status'))
            with status_panel:
                clear_output(wait=True)
                display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyimpan konfigurasi strategi pelatihan..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui(ui_components)
                
                # Simpan konfigurasi
                success = config_manager.save_module_config('training_strategy', updated_config)
                
                # Pastikan UI components teregistrasi untuk persistensi
                try:
                    config_manager.register_ui_components('training_strategy', ui_components)
                except Exception as persist_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
                
                # Tampilkan pesan sukses atau warning
                status_panel = ui_components.get('status_panel', ui_components.get('status'))
                with status_panel:
                    clear_output(wait=True)
                    if success:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disimpan",
                            alert_type='success'
                        ))
                    else:
                        display(create_info_alert(
                            f"{ICONS.get('warning', '⚠️')} Konfigurasi strategi pelatihan mungkin tidak tersimpan ke file",
                            alert_type='warning'
                        ))
                
                # Update info panel jika ada
                update_training_strategy_info(ui_components)
                
                # Sinkronisasi ke Google Drive jika diaktifkan
                try:
                    if env.is_drive_mounted:
                        # Sinkronisasi ke Google Drive
                        logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi strategi pelatihan ke Google Drive...")
                        sync_to_drive(None, ui_components)
                except Exception as e:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disimpan")
            except Exception as e:
                status_panel = ui_components.get('status_panel', ui_components.get('status'))
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Handler untuk tombol reset
        def on_reset_click(b):
            status_panel = ui_components.get('status_panel', ui_components.get('status'))
            with status_panel:
                clear_output(wait=True)
                display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Mereset konfigurasi strategi pelatihan..."))
            
            try:
                # Dapatkan default config
                default_config = get_default_config()
                
                # Update UI dari default config
                update_ui_from_config(ui_components, default_config)
                
                # Simpan default config
                success = config_manager.save_module_config('training_strategy', default_config)
                
                # Pastikan UI components teregistrasi untuk persistensi
                try:
                    config_manager.register_ui_components('training_strategy', ui_components)
                except Exception as persist_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
                
                # Tampilkan pesan sukses atau warning
                status_panel = ui_components.get('status_panel', ui_components.get('status'))
                with status_panel:
                    clear_output(wait=True)
                    if success:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil direset ke default",
                            alert_type='success'
                        ))
                    else:
                        display(create_info_alert(
                            f"{ICONS.get('warning', '⚠️')} Konfigurasi strategi pelatihan direset di UI tetapi mungkin tidak tersimpan ke file",
                            alert_type='warning'
                        ))
                
                # Update info panel jika ada
                update_training_strategy_info(ui_components)
                
                # Sinkronisasi ke Google Drive jika diaktifkan
                try:
                    if env.is_drive_mounted:
                        # Sinkronisasi ke Google Drive
                        logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi strategi pelatihan ke Google Drive...")
                        sync_to_drive(None, ui_components)
                except Exception as e:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil direset ke default")
            except Exception as e:
                status_panel = ui_components.get('status_panel', ui_components.get('status'))
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi: {str(e)}")
        
        # Pasang handler ke tombol
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(on_save_click)
        
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan handler ke ui_components
        ui_components.update({
            'on_save_click': on_save_click,
            'on_reset_click': on_reset_click
        })
        
        # Sinkronisasi otomatis dari Google Drive saat inisialisasi jika tersedia
        try:
            if env.is_drive_mounted:
                logger.info(f"{ICONS.get('info', 'ℹ️')} Memeriksa konfigurasi dari Google Drive...")
                # Cek file konfigurasi di Google Drive
                drive_config_file = os.path.join(env.drive_path, 'smartcash', 'configs', 'training_config.yaml')
                if os.path.exists(drive_config_file):
                    # Load konfigurasi dari Google Drive
                    sync_from_drive(None, ui_components)
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal memeriksa konfigurasi dari Google Drive: {str(e)}")
            # Tidak perlu menampilkan pesan error ke pengguna karena ini hanya pemeriksaan awal
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup button handlers: {str(e)}")
        return ui_components
