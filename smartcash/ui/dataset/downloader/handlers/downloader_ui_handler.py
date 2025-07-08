"""
Downloader UI Handler

Main UI handler for the downloader module following the standard module structure.
Inherits from core/ui_handler.py as per documentation.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.ui_handler import UIHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors


class DownloaderUIHandler(UIHandler):
    """Main UI handler for the downloader module."""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the downloader UI handler.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='downloader_ui_handler',
            parent_module='downloader',
            ui_components=ui_components,
            **kwargs
        )
    
    @handle_ui_errors(error_component_title="Downloader UI Handler Error", log_error=True)
    def initialize(self) -> None:
        """Initialize the downloader UI handler."""
        super().initialize()
        self.logger.info("🚀 Initializing Downloader UI handler")
        
        # Initialize operation manager
        from ..operations import DownloaderOperationManager
        self.operation_manager = DownloaderOperationManager(
            config=self.config,
            ui_components=self.ui_components,
            operation_container=self.ui_components.get('operation_manager')
        )
        
        # Setup button handlers
        self._setup_button_handlers()
        
        self.logger.info("✅ Downloader UI handler initialization complete")
    
    def _setup_button_handlers(self) -> None:
        """Setup button event handlers."""
        from .download_handler import setup_download_handlers
        
        # Setup download handlers with operation manager
        setup_download_handlers(
            ui_components=self.ui_components,
            config=self.config
        )
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Get UI components dictionary.
        
        Returns:
            Dictionary of UI components
        """
        return self.ui_components
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration and refresh UI.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        if hasattr(self, 'operation_manager'):
            self.operation_manager.config = self.config