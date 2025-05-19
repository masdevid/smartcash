"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Initializer untuk konfigurasi environment dengan alur otomatis yang lebih robust
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
from smartcash.ui.utils.ui_logger import create_direct_ui_logger

async def _sync_configs(config_manager: Any, ui_components: Dict[str, Any]) -> None:
    """
    Sinkronkan konfigurasi dengan Google Drive dan Colab
    
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
        
        # Sinkronkan dengan Google Drive
        await config_manager.sync_with_drive()
        ui_components['progress_bar'].value = 50
        
        # Sinkronkan dengan Colab jika dalam environment Colab
        if config_manager.is_colab_environment():
            await config_manager.sync_with_colab()
            ui_components['progress_bar'].value = 75
        
        # Update status
        ui_components['progress_bar'].value = 100
        ui_components['progress_message'].value = "Sinkronisasi selesai"
        logger.info("Konfigurasi berhasil disinkronkan")
        
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
    base_dir = str(Path.home() / '.smartcash')
    config_file = str(Path(base_dir) / 'config.yaml')
    config_manager = get_config_manager(base_dir=base_dir, config_file=config_file)
    
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
