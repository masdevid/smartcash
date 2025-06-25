"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer yang mewarisi CommonInitializer dengan clean dependency
"""

from typing import Dict, Any, List, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers

class DownloaderInitializer(CommonInitializer):
    """
    Downloader initializer with complete UI and backend service integration.
    
    This initializer handles the setup of the downloader UI components and their
    associated handlers, following the CommonInitializer pattern.
    """
    
    def __init__(self, parent_module: str = 'dataset'):
        """
        Initialize the downloader initializer.
        
        Args:
            parent_module: The parent module name (default: 'dataset')
        """
        self.parent_module = parent_module
        super().__init__(
            module_name=f"{parent_module}.downloader",
            config_handler_class=DownloaderConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create and return downloader UI components.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments (may include 'env' for environment)
            
        Returns:
            Dictionary of UI components
        """
        env = kwargs.get('env')
        ui_components = create_downloader_main_ui(config)
        
        # Add metadata and configuration to UI components
        ui_components.update({
            'downloader_initialized': True,
            'module_name': 'downloader',
            'parent_module': self.parent_module,
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'target_dir': config.get('download', {}).get('target_dir', 'data'),
            'env': env
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Set up the downloader handlers.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            **kwargs: Additional arguments (may include 'env' for environment)
            
        Returns:
            Updated dictionary of UI components with handlers attached
        """
        env = kwargs.get('env')
        return setup_download_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for the downloader.
        
        Returns:
            Default configuration dictionary
        """
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()
    
    def _get_critical_components(self) -> List[str]:
        """
        Get the list of critical component names that must be present in the UI.
        
        Returns:
            List of critical component names
        """
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'progress_container', 'show_for_operation', 
            'update_progress', 'complete_operation', 'error_operation', 'reset_all'
        ]
    
    def _pre_initialize_checks(self, **kwargs) -> None:
        """
        Perform any pre-initialization checks.
        
        Args:
            **kwargs: Additional arguments that might be needed for checks
            
        Raises:
            Exception: If any pre-initialization check fails
        """
        # Add any specific pre-initialization checks here
        pass

# Global instance and public API
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """
    Initialize and return the downloader UI.
    
    Args:
        env: Optional environment configuration
        config: Optional configuration overrides
        **kwargs: Additional arguments passed to the initializer
        
    Returns:
        The root UI component or error UI if initialization fails
    """
    return _downloader_initializer.initialize(env=env, config=config, **kwargs)