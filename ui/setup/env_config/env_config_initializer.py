"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Initializer untuk konfigurasi environment dengan dukungan Google Colab
"""

from typing import Dict, Any, Optional
from IPython.display import display
import asyncio
from pathlib import Path

from smartcash.ui.setup.env_config.components.env_config_component import create_env_config_ui
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
from smartcash.ui.setup.env_config.handlers.auto_check_handler import setup_auto_check_handler
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.common.config.singleton import Singleton
from smartcash.ui.utils.ui_logger import create_direct_ui_logger
from smartcash.common.constants.paths import COLAB_PATH
from smartcash.common.constants.core import DEFAULT_CONFIG_DIR, APP_NAME

async def _sync_configs(config_manager: Any, ui_components: Dict[str, Any]) -> None:
    """
    Sinkronkan konfigurasi dengan Google Drive
    
    Args:
        config_manager: Config manager instance
        ui_components: Dictionary UI components
    """
    logger = ui_components['logger']
    
    try:
        # Update progress
        ui_components['progress_message'].value = "Sinkronisasi konfigurasi..."
        ui_components['progress_message'].layout.visibility = 'visible'
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_bar'].value = 25
        
        # Sinkronkan semua konfigurasi dengan Google Drive
        results = config_manager.sync_all_configs()
        
        # Update progress berdasarkan hasil
        success_count = sum(1 for success, _ in results.values() if success)
        total_count = len(results)
        
        if success_count == total_count:
            ui_components['progress_bar'].value = 100
            ui_components['progress_message'].value = "Sinkronisasi selesai"
            logger.info("Semua konfigurasi berhasil disinkronkan")
        else:
            failed_files = [f for f, (success, _) in results.items() if not success]
            ui_components['progress_message'].value = f"Error: Gagal sinkronisasi {len(failed_files)} file"
            logger.error(f"Gagal sinkronisasi file: {', '.join(failed_files)}")
        
    except Exception as e:
        logger.error(f"Gagal sinkronisasi konfigurasi: {str(e)}")
        ui_components['progress_message'].value = f"Error: {str(e)}"
    finally:
        # Cleanup UI
        _cleanup_ui(ui_components)

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk konfigurasi environment.
    
    Returns:
        Dictionary UI components
    """
    # Inisialisasi environment manager
    env_manager = get_environment_manager()
    
    # Inisialisasi config manager dengan path yang sesuai
    if Singleton.is_colab_environment():
        # Gunakan path Colab
        base_dir = Path(COLAB_PATH) / APP_NAME
        config_file = 'base_config.yaml'  # File akan dicari di DEFAULT_CONFIG_DIR
    else:
        # Gunakan path lokal
        base_dir = Path.home() / APP_NAME
        config_file = 'base_config.yaml'  # File akan dicari di DEFAULT_CONFIG_DIR
    
    # Pastikan direktori ada
    base_dir.mkdir(parents=True, exist_ok=True)
    config_dir = base_dir / DEFAULT_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_manager = get_config_manager(
        base_dir=str(base_dir),
        config_file=config_file
    )
    
    # Buat komponen UI
    ui_components = create_env_config_ui(env_manager, config_manager)
    
    # Setup logger
    logger = create_direct_ui_logger(ui_components, "env_config")
    ui_components['logger'] = logger
    
    # Setup handlers
    setup_env_config_handlers(ui_components, env_manager, config_manager)
    
    # Setup auto check handler
    setup_auto_check_handler(ui_components)
    
    # Jalankan sinkronisasi otomatis
    asyncio.create_task(_sync_configs(config_manager, ui_components))
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components

def _disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Nonaktifkan UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary UI components
        disable: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang akan dinonaktifkan
    button_keys = ['drive_button', 'directory_button', 'check_button', 'save_button']
    
    # Nonaktifkan tombol
    for key in button_keys:
        if key in ui_components:
            ui_components[key].disabled = disable

def _cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan UI setelah proses selesai.
    
    Args:
        ui_components: Dictionary UI components
    """
    # Aktifkan kembali tombol
    _disable_ui_during_processing(ui_components, False)
    
    # Sembunyikan progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Sembunyikan progress message
    if 'progress_message' in ui_components:
        ui_components['progress_message'].layout.visibility = 'hidden'
