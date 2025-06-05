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
        """Create UI components dengan one-liner import dan direct creation"""
        from smartcash.ui.dataset.downloader.components.ui_layout import create_downloader_ui
        return create_downloader_ui(config, env)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan one-liner call"""
        return setup_download_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan one-liner"""
        return get_default_download_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components dengan one-liner list"""
        return ['ui', 'form_container', 'save_button', 'reset_button', 'download_button', 'check_button', 'cleanup_button', 'log_output', 'confirmation_area']

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

# One-liner utilities dengan proper argument handling
debug_layout_order = lambda ui_components: [type(child).__name__ for child in getattr(ui_components.get('ui'), 'children', [])] if ui_components and 'ui' in ui_components else ['No UI found']
debug_component_count = lambda ui_components: len([k for k in ui_components.keys() if not k.startswith('_')]) if ui_components else 0
debug_button_states = lambda ui_components: {k: getattr(v, 'disabled', 'N/A') for k, v in ui_components.items() if 'button' in k} if ui_components else {}
get_layout_summary = lambda ui_components: f"Components: {debug_component_count(ui_components)} | Layout: {len(debug_layout_order(ui_components))} widgets | Fixed: {ui_components.get('layout_order_fixed', False)}" if ui_components else "UI components not available"