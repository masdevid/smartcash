"""
File: smartcash/ui/dataset/download/download_init.py
Deskripsi: Refactored download initializer dengan ConfigHandler integration dan struktur yang lebih clean
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.download.handlers.download_config_handler import DownloadConfigHandler
from smartcash.ui.dataset.download.components.download_ui_factory import create_download_main_ui
from smartcash.ui.dataset.download.handlers.download_handlers_setup import setup_download_handlers
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE

class DownloadInitializer(CommonInitializer):
    """Refactored download initializer dengan ConfigHandler integration yang clean."""
    
    def __init__(self):
        super().__init__('download', DownloadConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan factory pattern."""
        return create_download_main_ui(config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan consolidated approach."""
        return setup_download_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration untuk download module."""
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022', 
            'version': '3',
            'organize_dataset': True,
            'backup_before_download': False
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical component keys yang harus ada."""
        return ['ui', 'main_container', 'download_button', 'check_button', 'save_button', 'reset_button']

# Global instance
_download_initializer = DownloadInitializer()

# Public API
def initialize_download_ui(env=None, config=None, **kwargs):
    """Initialize download UI dengan refactored approach."""
    return _download_initializer.initialize(env=env, config=config, **kwargs)

def get_download_config():
    """Get current download configuration."""
    return _download_initializer.get_current_config()

def get_download_status():
    """Get download module status."""
    return _download_initializer.get_module_status()