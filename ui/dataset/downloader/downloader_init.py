"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: One-liner downloader initializer tanpa UI fallbacks berlebihan
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.logger_bridge import get_logger
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloadConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config

MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(DOWNLOAD_LOGGER_NAMESPACE, 'DOWNLOAD')

class DownloadInitializer(CommonInitializer):
    """One-liner download initializer tanpa excessive fallbacks"""
    
    def __init__(self):
        super().__init__('downloader', DownloadConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan direct import - no fallbacks"""
        from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
        return create_downloader_main_ui(config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan one-liner call"""
        return setup_download_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan one-liner"""
        return get_default_download_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components dengan one-liner list"""
        return ['ui', 'form_container', 'save_button', 'reset_button', 'download_button', 'check_button', 'cleanup_button', 'log_output', 'confirmation_area', 
                'progress_tracker', 'progress_container', 'show_for_operation', 'update_progress', 'complete_operation', 'error_operation', 'reset_all']

# Global instance
_downloader_initializer = DownloadInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs) -> Any:
    """Initialize downloader UI dengan one-liner call"""
    return _downloader_initializer.initialize(env=env, config=config, **kwargs)

def get_downloader_config() -> Dict[str, Any]:
    """Get current downloader config dengan one-liner"""
    return getattr(getattr(_downloader_initializer, 'config_handler', None), 'get_current_config', lambda: {})()

def validate_downloader_layout(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate downloader layout dengan one-liner"""
    return {'valid': False, 'message': 'UI components tidak ditemukan - panggil initialize_downloader_ui() terlebih dahulu'} if not ui_components else _downloader_initializer.validate_layout_order(ui_components)

def get_downloader_status() -> Dict[str, Any]:
    """Get downloader status dengan one-liner update"""
    status = _downloader_initializer.get_module_status()
    status.update({'layout_order_fixed': True, 'current_config': get_downloader_config(), 'critical_components_count': len(_downloader_initializer._get_critical_components())})
    return status

def reset_downloader_layout() -> bool:
    """Reset downloader layout dengan one-liner global reassignment"""
    global _downloader_initializer
    _downloader_initializer = DownloadInitializer()
    return True
