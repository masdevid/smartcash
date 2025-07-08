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
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui_components
from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.operations.manager import DownloaderOperationManager
from smartcash.ui.dataset.downloader.operations.download_manager import DownloadHandlerManager
from smartcash.ui.core.errors.handlers import create_error_response

class DownloaderInitializer(ModuleInitializer):
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
        """Create downloader UI components following colab/dependency pattern

        Args:
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating downloader UI components with operation container")
            ui_components = create_downloader_ui_components(config, **kwargs)

            # Add metadata following the established pattern
            ui_components.update({
                'downloader_initialized': True,
                'module_name': 'downloader',
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'target_dir': config.get('download', {}).get('target_dir', 'data'),
                'logger': self.logger,
                'config': config,
                'env': env
            })

            self.logger.info(f"✅ UI components created successfully: {len(ui_components)} components")
            return ui_components
        except Exception as e:
            self.handle_error(f"Failed to create UI components: {str(e)}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI downloader")

    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers following colab/dependency pattern with OperationHandler

        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Updated UI components with handlers
        """
        try:
            self.logger.info("🔧 Setting up downloader operation manager")

            # Get operation container from UI components and set up operation handler
            operation_container = ui_components.get('operation_manager')
            if operation_container:
                # Create operation manager with operation container
                operation_manager = DownloaderOperationManager(
                    config=config,
                    operation_container=operation_container
                )
                
                # Store UI components reference in operation manager
                operation_manager._ui_components = ui_components
                
                # Initialize the operation manager
                operation_manager.initialize()
                
                # Store operation manager in UI components
                ui_components['downloader_operation_manager'] = operation_manager
                
                self.logger.info("✅ Downloader operation manager setup complete")
            else:
                self.logger.warning("⚠️ Operation container not found, using legacy handler")
                
                # Fallback to legacy handler
                handler_manager = DownloadHandlerManager(ui_components, config, env)
                handlers = handler_manager.setup_handlers()
                
                if handlers:
                    ui_components.update(handlers)

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
            from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
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

    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implementation of the abstract _initialize_impl method.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        # Call the parent implementation which handles the full initialization flow
        return super().initialize(**kwargs)

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