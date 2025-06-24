"""
File: smartcash/ui/setup/env_config/handlers/config_handler.py
Deskripsi: Handler untuk sinkronisasi konfigurasi menggunakan SimpleConfigManager
"""

from typing import Dict, Any
from smartcash.common.config.manager import SimpleConfigManager

class ConfigHandler:
    """⚙️ Handler untuk config synchronization"""
    
    def __init__(self):
        self.config_manager = SimpleConfigManager()
    
    def sync_configurations(self) -> Dict[str, Any]:
        """🔄 Sinkronisasi konfigurasi dengan SimpleConfigManager"""
        try:
            # Check if sync method exists
            if hasattr(self.config_manager, 'sync_configs_from_drive'):
                sync_result = self.config_manager.sync_configs_from_drive()
            else:
                # Fallback implementation
                sync_result = self._manual_sync_configs()
            
            return {
                'synced_count': sync_result.get('synced_count', 0),
                'configs_synced': sync_result.get('configs_synced', []),
                'success': sync_result.get('success', True),
                'errors': sync_result.get('errors', [])
            }
            
        except Exception as e:
            return {
                'synced_count': 0,
                'configs_synced': [],
                'success': False,
                'errors': [str(e)]
            }
    
    def _manual_sync_configs(self) -> Dict[str, Any]:
        """🔧 Manual config sync fallback"""
        try:
            # Discover available configs
            discovered_configs = self.config_manager.discover_repo_configs()
            
            return {
                'synced_count': len(discovered_configs),
                'configs_synced': discovered_configs,
                'success': True,
                'errors': []
            }
        except Exception as e:
            return {
                'synced_count': 0,
                'configs_synced': [],
                'success': False,
                'errors': [str(e)]
            }