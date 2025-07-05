"""
File: smartcash/ui/dataset/preprocessing/operations/base_operation.py
Deskripsi: Base operation handler untuk preprocessing module.
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.decorators.ui_decorators import safe_ui_operation


class BaseOperationHandler(ModuleUIHandler, ABC):
    """Base operation handler untuk preprocessing module.
    
    Features:
    - 🎯 Complete UI integration dengan core architecture
    - 📊 Progress tracking
    - 🔄 Button state management
    - 🔍 Confirmation dialogs
    - 📝 Summary reporting
    """
    
    def __init__(self, 
                 ui_components: Optional[Dict[str, Any]] = None,
                 module_name: str = "preprocessing", 
                 parent_module: str = "dataset"):
        """Initialize base operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
            module_name: Nama module
            parent_module: Parent module
        """
        super().__init__(module_name, parent_module)
        
        # Store UI components reference
        self.ui_components = ui_components or {}
        
        # Operation name (to be set by subclasses)
        self.operation_name = "base"
        
        # Button name (to be set by subclasses)
        self.button_name = None
        
        # Confirmation message (to be set by subclasses)
        self.confirmation_message = "Apakah Anda yakin ingin melanjutkan operasi ini?"
        
        # Success message (to be set by subclasses)
        self.success_message = "Operasi berhasil diselesaikan"
        
        # Failure message (to be set by subclasses)
        self.failure_message = "Operasi gagal diselesaikan"
        
        # Waiting message (to be set by subclasses)
        self.waiting_message = "Menunggu konfirmasi..."
        
        self.logger.debug(f"🎯 {self.__class__.__name__} initialized")
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components reference.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.ui_components = ui_components
    
    @safe_ui_operation(error_title="Operation Error")
    def execute(self) -> Dict[str, Any]:
        """Execute operation dengan confirmation dialog dan error handling.
        
        Returns:
            Operation result dictionary
        """
        # Disable button during operation
        self._set_button_enabled(False)
        
        # Show waiting message
        self.update_status(self.waiting_message)
        
        # Show confirmation dialog
        confirmed = self.show_confirmation_dialog(
            title=f"{self.operation_name.capitalize()} Confirmation",
            message=self.confirmation_message
        )
        
        if not confirmed:
            self.logger.info(f"🚫 {self.operation_name} operation cancelled by user")
            self._set_button_enabled(True)
            self.update_status(f"{self.operation_name.capitalize()} dibatalkan")
            return {"status": False, "cancelled": True}
        
        try:
            # Reset progress
            self.reset_progress()
            
            # Execute operation
            self.logger.info(f"🚀 Executing {self.operation_name} operation")
            result = self._execute_operation()
            
            # Update status and summary
            if result.get("status", False):
                self.update_status(self.success_message, "success")
                self.update_summary(self.success_message, "success", f"{self.operation_name.capitalize()} Success")
            else:
                error_message = result.get("error", self.failure_message)
                self.update_status(error_message, "error")
                self.update_summary(error_message, "error", f"{self.operation_name.capitalize()} Failed")
            
            return result
            
        except Exception as e:
            error_message = f"Error during {self.operation_name}: {str(e)}"
            self.logger.error(f"❌ {error_message}")
            self.update_status(error_message, "error")
            self.update_summary(error_message, "error", f"{self.operation_name.capitalize()} Error")
            return {"status": False, "error": str(e)}
            
        finally:
            # Re-enable button
            self._set_button_enabled(True)
    
    @abstractmethod
    def _execute_operation(self) -> Dict[str, Any]:
        """Execute actual operation logic.
        
        Returns:
            Operation result dictionary with at least a 'status' key
        """
        pass
    
    def _set_button_enabled(self, enabled: bool) -> None:
        """Set operation button enabled/disabled state.
        
        Args:
            enabled: Whether to enable the button
        """
        if not self.button_name or not self.ui_components:
            return
            
        button = self.ui_components.get(self.button_name)
        if button:
            button.disabled = not enabled
    
    def update_status(self, message: str, status_type: str = "info") -> None:
        """Update status panel dengan message.
        
        Args:
            message: Message to display
            status_type: Status type (info, success, warning, error)
        """
        if not self.ui_components:
            self.logger.warning("⚠️ UI components tidak tersedia untuk update status")
            return
            
        status_panel = self.ui_components.get('status_panel')
        if not status_panel:
            self.logger.warning("⚠️ Status panel tidak tersedia")
            return
            
        # Use parent class method for status panel update
        self.update_status_panel(status_panel, message, status_type)
    
    def update_summary(self, message: str, status_type: str = "info", title: str = None) -> None:
        """Update summary container dengan message.
        
        Args:
            message: Message to display
            status_type: Status type (info, success, warning, error)
            title: Optional title for summary
        """
        if not self.ui_components:
            self.logger.warning("⚠️ UI components tidak tersedia untuk update summary")
            return
            
        summary_container = self.ui_components.get('summary_container')
        if not summary_container:
            self.logger.warning("⚠️ Summary container tidak tersedia")
            return
            
        # Use parent class method for status panel update
        self.update_status_panel(summary_container, message, status_type, title)
    
    def update_progress(self, current: int, total: int, message: str = None) -> None:
        """Update progress tracker.
        
        Args:
            current: Current progress value
            total: Total progress value
            message: Optional progress message
        """
        if not self.ui_components:
            self.logger.warning("⚠️ UI components tidak tersedia untuk update progress")
            return
            
        progress_tracker = self.ui_components.get('progress_tracker')
        if not progress_tracker:
            self.logger.warning("⚠️ Progress tracker tidak tersedia")
            return
            
        # Calculate percentage
        percentage = int((current / total) * 100) if total > 0 else 0
        
        # Update progress tracker
        progress_tracker.value = percentage
        
        # Update progress message if provided
        if message and hasattr(progress_tracker, 'description'):
            progress_tracker.description = message
    
    def reset_progress(self) -> None:
        """Reset progress tracker to 0."""
        if not self.ui_components:
            return
            
        progress_tracker = self.ui_components.get('progress_tracker')
        if not progress_tracker:
            return
            
        # Reset progress tracker
        progress_tracker.value = 0
        
        # Reset description if available
        if hasattr(progress_tracker, 'description'):
            progress_tracker.description = ""
    
    def extract_config(self) -> Dict[str, Any]:
        """Extract config dari config handler.
        
        Returns:
            Configuration dictionary
        """
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.warning("⚠️ Config handler tidak tersedia")
            return {}
            
        # Get config dari handler
        if hasattr(config_handler, 'config'):
            return config_handler.config
        else:
            self.logger.warning("⚠️ Config handler tidak memiliki config property")
            return {}
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari config handler.
        
        Returns:
            Default configuration dictionary
        """
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.warning("⚠️ Config handler tidak tersedia")
            return {}
            
        # Get default config dari handler
        if hasattr(config_handler, 'get_default_config'):
            try:
                return config_handler.get_default_config()
            except Exception as e:
                self.logger.error(f"❌ Error getting default config: {str(e)}")
                return {}
        else:
            self.logger.warning("⚠️ Config handler tidak memiliki get_default_config method")
            return {}
