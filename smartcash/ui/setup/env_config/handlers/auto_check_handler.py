"""
File: smartcash/ui/setup/env_config/handlers/auto_check_handler.py
Deskripsi: Handler untuk auto check environment
"""

from typing import Dict, Any
import asyncio

from smartcash.common.config.singleton import Singleton

def setup_auto_check_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk auto check environment
    
    Args:
        ui_components: Dictionary UI components
    """
    logger = ui_components['logger']
    output = ui_components['output']
    
    async def auto_check():
        """
        Auto check environment saat startup
        """
        try:
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
                print("========================")
            
            # Update progress
            ui_components['progress_bar'].value = 100
            ui_components['progress_message'].value = "Pemeriksaan selesai"
            logger.info("Environment berhasil diperiksa")
            
        except Exception as e:
            logger.error(f"Gagal memeriksa environment: {str(e)}")
            ui_components['progress_message'].value = f"Error: {str(e)}"
    
    # Jalankan auto check
    asyncio.create_task(auto_check())
