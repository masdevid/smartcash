"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk konfigurasi environment
"""

import ipywidgets as widgets
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
    import threading
    import time
    from smartcash.ui.utils.alert_utils import create_info_box
    
    def auto_check_and_sync():
        # Tunggu sebentar agar UI logger siap
        time.sleep(1)
        
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
            
            # Sinkronisasi dengan ConfigManager
            try:
                from smartcash.common.config.manager import get_config_manager
                config_manager = get_config_manager()
                
                # Pastikan konfigurasi di-reload untuk mendapatkan perubahan terbaru
                config_manager.reload()
                
                # Simpan konfigurasi untuk memastikan perubahan tersimpan
                config_manager.save()
                
                logger.debug("🔄 Konfigurasi berhasil di-reload dan disimpan")
            except Exception as config_error:
                logger.debug(f"ℹ️ Tidak dapat melakukan sinkronisasi ConfigManager: {str(config_error)}")
            
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
    
    # Jalankan pemeriksaan dan sinkronisasi di thread terpisah
    auto_thread = threading.Thread(target=auto_check_and_sync)
    auto_thread.daemon = True
    auto_thread.start()
    
    return ui_components

# Fungsi _disable_ui_during_processing dan _cleanup_ui dipindahkan ke ui_helpers.py
