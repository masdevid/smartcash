"""
File: smartcash/ui/setup/env_config/handlers/config_handler.py
Deskripsi: Handler untuk sinkronisasi konfigurasi menggunakan ConfigManager
"""

from typing import Dict, Any
from smartcash.common.config.manager import ConfigManager

class ConfigHandler:
    """⚙️ Handler untuk config synchronization"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def sync_configurations(self) -> Dict[str, Any]:
        """🔄 Sinkronisasi konfigurasi dengan ConfigManager"""
        try:
            # Gunakan ConfigManager untuk sync
            sync_result = self.config_manager.sync_configs_from_drive()
            
            return {
                'synced_count': sync_result.get('synced_count', 0),
                'configs_synced': sync_result.get('configs_synced', []),
                'success': sync_result.get('success', False),
                'errors': sync_result.get('errors', [])
            }
            
        except Exception as e:
            return {
                'synced_count': 0,
                'configs_synced': [],
                'success': False,
                'errors': [str(e)]
            }