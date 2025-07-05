"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer yang mewarisi CommonInitializer dengan clean dependency

Initialization Flow:
1. Load and validate configuration
2. Create UI components
3. Setup module handlers
4. Return UI with proper error handling
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.operation import DownloadHandlerManager
from smartcash.ui.core.errors.handlers import create_error_response

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan complete UI dan backend service integration

    Provides a structured approach to initializing the dataset downloader module with
    proper error handling, logging, and UI component management. Follows the same
    initialization flow as CommonInitializer with additional downloader-specific
    functionality.
    """

    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler
        )

    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components dengan environment awareness

        Args:
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info(" Membuat komponen UI downloader")
            ui_components = create_downloader_main_ui(config)

            # Add metadata
            ui_components.update({
                'downloader_initialized': True,
                'module_name': 'downloader',
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'target_dir': config.get('download', {}).get('target_dir', 'data'),
                'logger': self.logger  # Ensure logger is available
            })

            self.logger.debug(f"UI components created: {list(ui_components.keys())}")
            return ui_components
        except Exception as e:
            self.handle_error(f"Failed to create UI components: {str(e)}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI downloader")

    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan backend service integration

        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Updated UI components with handlers
        """
        try:
            self.logger.info(" Menyiapkan handlers downloader")

            # Buat instance DownloadHandlerManager dan setup handlers
            handler_manager = DownloadHandlerManager(ui_components, config, env)
            handlers = handler_manager.setup_handlers()

            # Update UI components with handlers
            if handlers:
                ui_components.update(handlers)
                self.logger.debug(f"Handlers setup complete: {list(handlers.keys()) if handlers else 'No handlers'}")

            return ui_components
        except Exception as e:
            self.handle_error(f"Failed to setup module handlers: {str(e)}", exc_info=True)
            return ui_components  # Return original components to avoid breaking the UI

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk downloader

        Returns:
            Default configuration dictionary
        """
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
            default_config = get_default_downloader_config()
            self.logger.debug("Default config loaded successfully")
            return default_config
        except Exception as e:
            self.handle_error(f"Failed to get default config: {str(e)}", exc_info=True)
            # Return minimal working config to prevent crashes
            return {
                'data': {'dir': 'data'},
                'download': {'target_dir': 'data'}
            }

    def _get_critical_components(self) -> List[str]:
        """Get list of critical UI components that must exist

        Returns:
            List of critical component keys
        """
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button',
            'progress_tracker', 'log_output', 'status_panel',
        ]

    def pre_initialize_checks(self, **kwargs) -> None:
        """Perform pre-initialization checks

        Raises:
            Exception: If any pre-initialization check fails
        """
        # Check if we're in a supported environment
        try:
            import IPython
            # Additional checks can be added here
        except ImportError:
            raise RuntimeError("Dataset downloader requires IPython environment")

# Global instance dan public API
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk downloader UI dengan parent module support

    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components with 'ui' as the main component
    """
    return _downloader_initializer.initialize(config=config, env=env, **kwargs)