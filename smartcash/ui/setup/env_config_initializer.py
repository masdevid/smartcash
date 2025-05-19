"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk konfigurasi environment
"""

import ipywidgets as widgets
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from IPython.display import display

from smartcash.ui.setup.env_config_component import create_env_config_ui
from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.ui_logger import create_direct_ui_logger
from smartcash.ui.setup.ui_helpers import disable_ui_during_processing, cleanup_ui

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi environment
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Inisialisasi environment manager
    env_manager = get_environment_manager()
    
    # Inisialisasi config manager
    config_manager = get_config_manager()
    
    # Buat komponen UI
    ui_components = create_env_config_ui(env_manager, config_manager)
    
    # Setup logger
    logger = create_direct_ui_logger(ui_components, "env_config")
    ui_components['logger'] = logger
    
    # Setup handlers
    setup_env_config_handlers(ui_components, env_manager, config_manager)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    # Hapus log info inisialisasi
    
    # Otomatisasi pemeriksaan environment dan sinkronisasi konfigurasi
    import time
    from smartcash.ui.utils.alert_utils import create_info_box
    
    def auto_check_and_sync():
        # Tidak perlu menunggu karena tidak menggunakan threading
        # Hanya berikan sedikit jeda agar UI terender terlebih dahulu
        time.sleep(0.5)
        
        try:
            # Update status panel untuk menunjukkan proses sedang berjalan
            ui_components['status_panel'].value = create_info_box(
                "Pemeriksaan Environment", 
                "Sedang memeriksa environment dan melakukan sinkronisasi konfigurasi...",
                style="info"
            ).value
            
            # Periksa environment tanpa log berlebihan
            env_info = env_manager.get_system_info()
            
            # Pastikan direktori configs ada
            configs_dir = Path(env_manager.base_dir) / 'configs'
            os.makedirs(configs_dir, exist_ok=True)
            
            # Sinkronisasi konfigurasi dengan Google Drive jika terhubung
            if hasattr(env_manager, 'sync_config'):
                sync_success, sync_message = env_manager.sync_config()
                if not sync_success:
                    logger.warning(f"⚠️ {sync_message}")
            
            # Simpan konfigurasi environment
            if hasattr(env_manager, 'save_environment_config'):
                save_success, save_message = env_manager.save_environment_config()
                if not save_success:
                    logger.warning(f"⚠️ {save_message}")
            
            # Sinkronisasi semua file konfigurasi di direktori configs
            try:
                # Import fungsi sinkronisasi dari smartcash.common.config.sync
                from smartcash.common.config.sync import sync_all_configs
                
                # Tampilkan status di log output
                ui_components['status_panel'].value = create_info_box(
                    "Sinkronisasi Konfigurasi", 
                    "Sedang menyinkronkan semua file konfigurasi...",
                    style="info"
                ).value
                
                # Sinkronisasi semua file konfigurasi tanpa log berlebihan
                results = sync_all_configs(
                    sync_strategy='merge',  # Gabungkan konfigurasi lokal dan drive
                    config_dir='configs',   # Direktori konfigurasi
                    create_backup=True,     # Buat backup sebelum sinkronisasi
                    logger=None            # Tidak perlu log detail proses
                )
                
                # Log hasil sinkronisasi ke UI secara minimal
                counts = {k: len(v) for k, v in results.items()}
                
                # Tampilkan hasil hanya untuk file *_config.yaml yang berhasil
                if counts['success'] > 0:
                    success_files = [item['file'] for item in results['success'] if '_config.yaml' in item['file']]
                    if success_files:
                        logger.info(f"✅ {len(success_files)} file konfigurasi berhasil disinkronkan: {', '.join(success_files)}")
                
                # Log error jika ada
                if counts['failure'] > 0:
                    # Hanya tampilkan jumlah error, bukan detail setiap file
                    logger.warning(f"⚠️ {counts['failure']} file gagal disinkronkan")
                
                # Update status panel dengan hasil
                ui_components['status_panel'].value = create_info_box(
                    "Sinkronisasi Konfigurasi", 
                    f"Sinkronisasi selesai: {counts['success']} disinkronkan, {counts['skipped']} dilewati, {counts['failure']} gagal",
                    style="success" if counts['failure'] == 0 else "warning"
                ).value
            except Exception as config_error:
                # Log error ke UI
                logger.error(f"❌ Error saat sinkronisasi konfigurasi: {str(config_error)}")
                
                # Update status panel dengan error
                ui_components['status_panel'].value = create_info_box(
                    "Error Sinkronisasi", 
                    f"Terjadi kesalahan saat sinkronisasi konfigurasi: {str(config_error)}",
                    style="error"
                ).value
            
            # Update status panel dengan hasil
            ui_components['status_panel'].value = create_info_box(
                "Konfigurasi Environment", 
                "Pemeriksaan environment dan sinkronisasi konfigurasi berhasil dilakukan.",
                style="success"
            ).value
            
            # Nonaktifkan tombol Drive jika sudah terhubung
            if env_manager.is_drive_mounted:
                ui_components['drive_button'].disabled = True
                ui_components['drive_button'].description = "Drive Terhubung"
                ui_components['drive_button'].tooltip = "Google Drive sudah terhubung"
                ui_components['drive_button'].icon = "check"
            
            # Log ringkasan
            logger.info("✅ Pemeriksaan environment dan sinkronisasi konfigurasi berhasil")
        except Exception as e:
            # Update status panel dengan error
            ui_components['status_panel'].value = create_info_box(
                "Error", 
                f"Terjadi kesalahan saat pemeriksaan environment: {str(e)}",
                style="error"
            ).value
            logger.error(f"❌ Error saat otomatisasi: {str(e)}")
    
    # Jalankan pemeriksaan dan sinkronisasi secara langsung
    # Colab tidak bekerja baik dengan threading
    auto_check_and_sync()
    
    return ui_components

# Fungsi _disable_ui_during_processing dan _cleanup_ui dipindahkan ke ui_helpers.py
