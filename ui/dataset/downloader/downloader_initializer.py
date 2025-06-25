"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer yang mewarisi CommonInitializer dengan clean dependency
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan complete UI dan backend service integration"""
    
    def __init__(self):
        self.parent_module = 'dataset'  # Store parent module as instance variable
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components dengan environment awareness"""
        ui_components = create_downloader_main_ui(config)
        ui_components.update({
            'downloader_initialized': True,
            'module_name': 'downloader',
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'target_dir': config.get('download', {}).get('target_dir', 'data')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan backend service integration"""
        return setup_download_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk downloader"""
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'progress_container', 'show_for_operation', 
            'update_progress', 'complete_operation', 'error_operation', 'reset_all'
        ]

# Global instance dan public API
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Factory function untuk downloader UI dengan parent module support"""
    return _downloader_initializer.initialize(env=env, config=config, **kwargs)