"""
File: smartcash/ui/dataset/downloader/operations/download_manager.py
Deskripsi: Manager untuk mengintegrasikan semua operation handlers
"""

from typing import Dict, Any, Optional, List, Callable
from smartcash.ui.core.errors.handlers import handle_ui_errors
from .manager import DownloaderOperationManager

class DownloadHandlerManager(DownloaderOperationManager):
    """Manager untuk mengintegrasikan semua operation handlers."""
    
    @handle_ui_errors(error_component_title="Download Handler Manager Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize download handler manager.
        
        Args:
            ui_components: Dictionary UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(config={}, ui_components=ui_components, **kwargs)
        
        # Initialize operation handlers
        self.initialize()
        
        # Button references for enabling/disabling during operations
        self._buttons = {}
        self._get_button_references()
    
    def _get_button_references(self) -> None:
        """Get references to buttons for enabling/disabling during operations."""
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self.logger.warning("⚠️ No UI components available for button references")
            return
            
        # Get button references from UI components
        button_keys = ['download_button', 'check_button', 'cleanup_button']
        for key in button_keys:
            if key in self._ui_components:
                self._buttons[key] = self._ui_components[key]
                
        self.logger.debug(f"📋 Found {len(self._buttons)} button references")
    
    def disable_buttons(self) -> None:
        """Disable buttons during operation execution."""
        for button_name, button in self._buttons.items():
            if hasattr(button, 'disabled'):
                button.disabled = True
    
    def enable_buttons(self) -> None:
        """Enable buttons after operation execution."""
        for button_name, button in self._buttons.items():
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    def _reset_progress_tracker(self) -> None:
        """Reset progress tracker after operation."""
        if self.operation_container and hasattr(self.operation_container, 'reset_progress'):
            self.operation_container.reset_progress()
    
    @handle_ui_errors(error_component_title="Download Operation Error", log_error=True)
    def _execute_download_operation(self) -> None:
        """Execute download operation."""
        # Disable buttons during operation
        self.disable_buttons()
        
        # Execute download
        result = self.download_handler.execute_download()
        
        # Enable buttons after operation
        self.enable_buttons()
        
        # Clear progress tracker
        self._reset_progress_tracker()
    
    @handle_ui_errors(error_component_title="Check Operation Error", log_error=True)
    def _execute_check_operation(self) -> None:
        """Execute check operation."""
        # Disable buttons during operation
        self.disable_buttons()
        
        # Execute check
        result = self.check_handler.execute_check()
        
        # Enable buttons after operation
        self.enable_buttons()
        
        # Clear progress tracker
        self._reset_progress_tracker()
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error", log_error=True)
    def _execute_cleanup_operation(self, targets: List[str]) -> None:
        """Execute cleanup operation (dangerous, removes files).
        
        Args:
            targets: List of targets to clean up
        """
        # Disable buttons during operation
        self.disable_buttons()
        
        # Execute cleanup
        result = self.cleanup_handler.execute_cleanup(targets)
        
        # Enable buttons after operation
        self.enable_buttons()
        
        # Clear progress tracker
        self._reset_progress_tracker()
    
    def register_button_callbacks(self) -> None:
        """Register button callbacks for download, check, and cleanup operations."""
        try:
            # Register download button callback
            if 'download_button' in self._ui_components:
                self._ui_components['download_button'].on_click(
                    lambda _: self._execute_download_operation()
                )
                self.logger.info("✅ Registered download button callback")
            
            # Register check button callback
            if 'check_button' in self._ui_components:
                self._ui_components['check_button'].on_click(
                    lambda _: self._execute_check_operation()
                )
                self.logger.info("✅ Registered check button callback")
            
            # Register cleanup button callback with targets
            if 'cleanup_button' in self._ui_components and 'cleanup_targets' in self._ui_components:
                self._ui_components['cleanup_button'].on_click(
                    lambda _: self._execute_cleanup_operation(
                        self._ui_components['cleanup_targets'].value
                    )
                )
                self.logger.info("✅ Registered cleanup button callback")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to register button callbacks: {e}")
