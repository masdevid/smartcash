"""
File: smartcash/ui/setup/env_config/handlers/auto_check_handler.py
Deskripsi: Handler untuk pemeriksaan otomatis environment
"""

import time
from typing import Dict, Any
from tqdm.notebook import tqdm

from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.setup.env_config.handlers.sync_handler import sync_configs, run_sync_tests
from smartcash.ui.setup.env_config.utils.env_utils import get_env_status
from smartcash.common.environment import get_environment_manager

def auto_check_and_sync(ui_components: Dict[str, Any], env_manager: Any) -> None:
    """
    Otomatisasi pemeriksaan environment dan sinkronisasi konfigurasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env_manager: Environment manager
    """
    # Berikan sedikit jeda agar UI terender terlebih dahulu
    time.sleep(0.5)
    
    logger = ui_components['logger']
    
    try:
        # Update status panel untuk menunjukkan proses sedang berjalan
        ui_components['status_panel'].value = create_info_box(
            "Pemeriksaan Environment", 
            "Sedang memeriksa environment dan melakukan sinkronisasi konfigurasi...",
            style="info"
        ).value
        
        # Tampilkan progress bar untuk proses yang berjalan
        steps = ["Pemeriksaan Environment", "Sinkronisasi Konfigurasi", "Test Sinkronisasi", "Finalisasi"]
        with tqdm(total=len(steps), desc="Setup Environment") as pbar:
            # Langkah 1: Pemeriksaan Environment
            pbar.set_description("Pemeriksaan Environment")
            env_status = get_env_status(env_manager)
            logger.info(f"🔍 Sistem operasi: {env_status['system_info'].get('os', 'Tidak diketahui')}")
            logger.info(f"🐍 Python version: {env_status['system_info'].get('python_version', 'Tidak diketahui')}")
            logger.info(f"🔄 Google Drive: {'Terhubung' if env_status['drive_status'].get('is_mounted', False) else 'Tidak terhubung'}")
            pbar.update(1)
            
            # Langkah 2: Sinkronisasi Konfigurasi
            pbar.set_description("Sinkronisasi Konfigurasi")
            sync_configs(ui_components, env_manager, logger)
            pbar.update(1)
            
            # Langkah 3: Test Sinkronisasi
            pbar.set_description("Test Sinkronisasi")
            run_sync_tests(ui_components, logger)
            pbar.update(1)
            
            # Langkah 4: Finalisasi
            pbar.set_description("Finalisasi")
            update_ui_after_check(ui_components, env_manager)
            pbar.update(1)
        
        # Update status panel dengan hasil
        ui_components['status_panel'].value = create_info_box(
            "Konfigurasi Environment", 
            "Pemeriksaan environment dan sinkronisasi konfigurasi berhasil dilakukan.",
            style="success"
        ).value
        
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

def update_ui_after_check(ui_components: Dict[str, Any], env_manager: Any) -> None:
    """
    Update UI setelah pemeriksaan environment
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env_manager: Environment manager
    """
    # Nonaktifkan tombol Drive jika sudah terhubung
    if env_manager.is_drive_mounted:
        ui_components['drive_button'].disabled = True
        ui_components['drive_button'].description = "Drive Terhubung"
        ui_components['drive_button'].tooltip = "Google Drive sudah terhubung"
        ui_components['drive_button'].icon = "check"

def setup_auto_check_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk pemeriksaan otomatis environment
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Dapatkan environment manager
    env_manager = get_environment_manager()
    
    # Jalankan pemeriksaan otomatis secara asinkron
    from IPython.display import display
    import threading
    
    # Buat thread untuk menjalankan pemeriksaan otomatis
    def run_check():
        auto_check_and_sync(ui_components, env_manager)
    
    # Jalankan thread
    thread = threading.Thread(target=run_check)
    thread.daemon = True
    thread.start()
