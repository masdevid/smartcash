"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Optimized downloader initializer dengan streamlined components dan progress tracker
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloadConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui

MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(DOWNLOAD_LOGGER_NAMESPACE, 'DOWNLOAD')

class DownloadInitializer(CommonInitializer):
    """Optimized download initializer dengan streamlined components"""
    
    def __init__(self):
        super().__init__('downloader', DownloadConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create streamlined UI components"""
        ui_components = create_downloader_main_ui(config)
        
        if not ui_components or 'ui' not in ui_components:
            raise ValueError("UI components creation failed - missing 'ui' component")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan streamlined approach"""
        setup_download_handlers(ui_components, config, env)
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get streamlined default config"""
        return get_default_download_config()
    
    def _get_critical_components(self) -> List[str]:
        """Streamlined critical components list"""
        return [
            'ui', 'form_container', 'save_button', 'reset_button',
            'download_button', 'check_button', 'cleanup_button',
            'log_output', 'confirmation_area', 'progress_tracker', 'progress_container'
        ]

# Factory dan initialization functions
def create_downloader_initializer() -> DownloadInitializer:
    """Factory untuk downloader initializer"""
    return DownloadInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Initialize downloader UI dengan optimized error handling"""
    try:
        initializer = create_downloader_initializer()
        return initializer.initialize(env=env, config=config, **kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize downloader UI: {str(e)}")
        return None

# Utility functions dengan one-liner optimization
get_downloader_config = lambda: create_downloader_initializer().config_handler.get_default_config() if hasattr(create_downloader_initializer(), 'config_handler') else get_default_download_config()
reset_downloader_ui = lambda: True  # Next initialization akan create fresh instance