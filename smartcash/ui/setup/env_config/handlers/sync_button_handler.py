"""
File: smartcash/ui/setup/env_config/handlers/sync_button_handler.py
Deskripsi: Handler untuk tombol sinkronisasi konfigurasi
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

from smartcash.ui.setup.env_config.env_config_initializer import _sync_configs

def setup_sync_button_handler(ui_components: Dict[str, Any], config_manager: Any) -> None:
    """
    Setup handler untuk tombol sinkronisasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config_manager: Konfigurasi manager
    """
    logger = ui_components['logger']
    sync_button = ui_components['sync_button']
    status_panel = ui_components['status_panel']
    
    async def on_sync_button_clicked(b):
        """
        Handler untuk event klik tombol sinkronisasi
        """
        try:
            # Nonaktifkan tombol selama sinkronisasi
            sync_button.disabled = True
            
            # Jalankan sinkronisasi
            await _sync_configs(config_manager, ui_components)
            
            # Update status panel
            last_sync = config_manager.get_last_sync_time()
            sync_status = "Terakhir sinkronisasi: " + (last_sync.strftime("%Y-%m-%d %H:%M:%S") if last_sync else "Belum pernah")
            
            status_panel.value = f"""
            <div class="alert alert-info">
                <strong>Konfigurasi Environment</strong><br>
                Sistem akan melakukan pemeriksaan environment dan sinkronisasi konfigurasi secara otomatis.<br>
                {sync_status}
            </div>
            """
            
            logger.info("Sinkronisasi manual berhasil")
            
        except Exception as e:
            logger.error(f"Gagal sinkronisasi manual: {str(e)}")
        finally:
            # Aktifkan kembali tombol
            sync_button.disabled = False
    
    # Register click handler
    sync_button.on_click(lambda b: asyncio.create_task(on_sync_button_clicked(b))) 