"""
File: smartcash/ui/setup/env_config/handlers/setup_handlers.py
Deskripsi: Setup handler untuk konfigurasi environment
"""

from typing import Dict, Any
import asyncio
from pathlib import Path

from smartcash.common.environment import EnvironmentManager
from smartcash.common.config.manager import ConfigManager
from smartcash.common.config.singleton import Singleton

def setup_env_config_handlers(ui_components: Dict[str, Any], env_manager: EnvironmentManager, config_manager: ConfigManager) -> None:
    """
    Setup handler untuk konfigurasi environment
    
    Args:
        ui_components: Dictionary UI components
        env_manager: Environment manager instance
        config_manager: Config manager instance
    """
    logger = ui_components['logger']
    output = ui_components['output']
    
    async def on_sync_click(b):
        """
        Handler untuk tombol sinkronisasi
        """
        try:
            # Nonaktifkan tombol
            ui_components['sync_button'].disabled = True
            ui_components['check_button'].disabled = True
            ui_components['save_button'].disabled = True
            
            # Update progress
            ui_components['progress_message'].value = "Sinkronisasi konfigurasi..."
            ui_components['progress_message'].layout.visibility = 'visible'
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_bar'].value = 25
            
            # Sinkronkan dengan Google Drive
            success, message, _ = config_manager.sync_with_drive(config_manager.config_file)
            if not success:
                raise Exception(message)
            
            # Update progress
            ui_components['progress_bar'].value = 100
            ui_components['progress_message'].value = "Sinkronisasi selesai"
            logger.info("Konfigurasi berhasil disinkronkan")
            
        except Exception as e:
            logger.error(f"Gagal sinkronisasi konfigurasi: {str(e)}")
            ui_components['progress_message'].value = f"Error: {str(e)}"
        finally:
            # Aktifkan kembali tombol
            ui_components['sync_button'].disabled = False
            ui_components['check_button'].disabled = False
            ui_components['save_button'].disabled = False
    
    async def on_check_click(b):
        """
        Handler untuk tombol cek environment
        """
        try:
            # Nonaktifkan tombol
            ui_components['sync_button'].disabled = True
            ui_components['check_button'].disabled = True
            ui_components['save_button'].disabled = True
            
            # Update progress
            ui_components['progress_message'].value = "Memeriksa environment..."
            ui_components['progress_message'].layout.visibility = 'visible'
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_bar'].value = 25
            
            # Cek environment
            with output:
                output.clear_output()
                print("=== Status Environment ===")
                print(f"Environment: {'Colab' if Singleton.is_colab_environment() else 'Lokal'}")
                print(f"Base Directory: {config_manager.base_dir}")
                print(f"Config File: {config_manager.config_file}")
                print("========================")
            
            # Update progress
            ui_components['progress_bar'].value = 100
            ui_components['progress_message'].value = "Pemeriksaan selesai"
            logger.info("Environment berhasil diperiksa")
            
        except Exception as e:
            logger.error(f"Gagal memeriksa environment: {str(e)}")
            ui_components['progress_message'].value = f"Error: {str(e)}"
        finally:
            # Aktifkan kembali tombol
            ui_components['sync_button'].disabled = False
            ui_components['check_button'].disabled = False
            ui_components['save_button'].disabled = False
    
    async def on_save_click(b):
        """
        Handler untuk tombol simpan
        """
        try:
            # Nonaktifkan tombol
            ui_components['sync_button'].disabled = True
            ui_components['check_button'].disabled = True
            ui_components['save_button'].disabled = True
            
            # Update progress
            ui_components['progress_message'].value = "Menyimpan konfigurasi..."
            ui_components['progress_message'].layout.visibility = 'visible'
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_bar'].value = 25
            
            # Simpan konfigurasi
            config_manager.save_config()
            
            # Update progress
            ui_components['progress_bar'].value = 100
            ui_components['progress_message'].value = "Konfigurasi berhasil disimpan"
            logger.info("Konfigurasi berhasil disimpan")
            
        except Exception as e:
            logger.error(f"Gagal menyimpan konfigurasi: {str(e)}")
            ui_components['progress_message'].value = f"Error: {str(e)}"
        finally:
            # Aktifkan kembali tombol
            ui_components['sync_button'].disabled = False
            ui_components['check_button'].disabled = False
            ui_components['save_button'].disabled = False
    
    # Register handlers
    ui_components['sync_button'].on_click(lambda b: asyncio.create_task(on_sync_click(b)))
    ui_components['check_button'].on_click(lambda b: asyncio.create_task(on_check_click(b)))
    ui_components['save_button'].on_click(lambda b: asyncio.create_task(on_save_click(b)))
