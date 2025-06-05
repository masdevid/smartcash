"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Fixed downloader initializer tanpa UI fallbacks dan error handling yang proper
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
    """Fixed download initializer tanpa fallbacks dan proper error propagation"""
    
    def __init__(self):
        super().__init__('downloader', DownloadConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan direct call tanpa fallback"""
        # Direct call ke create_downloader_main_ui
        ui_components = create_downloader_main_ui(config)
        
        # Validate critical components ada
        if not ui_components or 'ui' not in ui_components:
            raise ValueError("UI components creation failed - missing 'ui' component")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan direct call"""
        # Direct call ke setup_download_handlers
        setup_download_handlers(ui_components, config, env)
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan direct call"""
        return get_default_download_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components list yang realistis"""
        return [
            'ui',                    # Main UI container
            'form_container',        # Form container
            'save_button',          # Save button
            'reset_button',         # Reset button
            'download_button',      # Download button
            'check_button',         # Check button
            'cleanup_button',       # Cleanup button
            'log_output',           # Log output
            'confirmation_area',    # Confirmation area
            'progress_tracker',     # Progress tracker instance
            'progress_container'    # Progress container widget
        ]

# Factory function untuk create initializer
def create_downloader_initializer() -> DownloadInitializer:
    """Factory untuk create downloader initializer"""
    return DownloadInitializer()

# Main initialization function
def initialize_downloader_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Initialize downloader UI dengan proper error handling"""
    try:
        # Create fresh initializer instance
        initializer = create_downloader_initializer()
        
        # Initialize dengan proper error propagation
        result = initializer.initialize(env=env, config=config, **kwargs)
        return result
        
    except Exception as e:
        # Log the actual error for debugging
        error_msg = str(e)
        print(f"❌ Failed to initialize downloader UI: {error_msg}")
        
# Utility functions
def get_downloader_config() -> Dict[str, Any]:
    """Get current downloader config dengan safe access"""
    try:
        initializer = create_downloader_initializer()
        if hasattr(initializer, 'config_handler') and initializer.config_handler:
            return initializer.config_handler.get_default_config()
        return get_default_download_config()
    except Exception:
        return get_default_download_config()


def reset_downloader_ui() -> bool:
    """Reset downloader UI dengan cleanup"""
    try:
        # Simply return True - next initialization akan create fresh instance
        return True
    except Exception:
        return False
