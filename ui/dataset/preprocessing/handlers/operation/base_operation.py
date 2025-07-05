"""
File: smartcash/ui/dataset/preprocessing/handlers/operation/base_operation.py
Deskripsi: Base operation handler untuk preprocessing operations
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.decorators.ui_decorators import safe_ui_operation
from smartcash.ui.dataset.preprocessing.handlers.base_preprocessing_handler import BasePreprocessingHandler

class BaseOperationHandler(BasePreprocessingHandler, ABC):
    """Base operation handler untuk preprocessing operations.
    
    Provides common functionality for all operation handlers:
    - Confirmation dialog management
    - Progress tracking
    - Button state management
    - Summary reporting
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize base operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components=ui_components)
        self._operation_name = "base"
        self._operation_title = "Base Operation"
        self._confirmation_pending = False
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute operation dengan proper error handling.
        
        Returns:
            Operation result
        """
        pass
    
    @handle_ui_errors(error_component_title="Operation Error", log_error=True)
    def _prepare_operation(self) -> bool:
        """Prepare operation execution dengan UI updates.
        
        Returns:
            True if preparation successful, False otherwise
        """
        # Clear previous outputs
        self.clear_ui_outputs(self.ui_components, self.get_output_keys())
        
        # Disable buttons during operation
        self.disable_all_buttons(self.ui_components, self.get_button_keys())
        
        # Setup progress tracking
        self._setup_progress()
        
        # Log operation start
        self.logger.info(f"🚀 Memulai {self._operation_name} operation")
        
        return True
    
    @handle_ui_errors(error_component_title="Operation Completion Error", log_error=True)
    def _complete_operation(self, success: bool = True, message: str = None) -> None:
        """Complete operation dengan UI updates.
        
        Args:
            success: Whether operation was successful
            message: Optional completion message
        """
        # Enable buttons after operation
        self.enable_all_buttons(self.ui_components, self.get_button_keys())
        
        # Complete progress tracking
        if success:
            self.complete_progress(self.ui_components)
            status_type = "success"
            icon = "✅"
        else:
            self.error_progress(self.ui_components)
            status_type = "error"
            icon = "❌"
        
        # Log operation completion
        completion_message = message or f"{self._operation_title} {'berhasil' if success else 'gagal'}"
        self.logger.info(f"{icon} {completion_message}")
        
        # Update summary panel
        if 'summary_container' in self.ui_components:
            self.update_summary(completion_message, status_type)
    
    def _setup_progress(self) -> None:
        """Setup progress tracking untuk operation."""
        progress_tracker = self.ui_components.get('progress_tracker')
        if not progress_tracker:
            self.logger.warning("⚠️ Progress tracker tidak tersedia")
            return
            
        self.update_progress(self.ui_components, 0, 100, f"Memulai {self._operation_name}...")
    
    def _create_progress_callback(self) -> Callable:
        """Create progress callback untuk API integration.
        
        Returns:
            Progress callback function
        """
        def progress_callback(level: str, current: int, total: int, message: str):
            """Update progress dengan proper error handling."""
            try:
                # Calculate percentage
                percentage = int((current / total) * 100) if total > 0 else 0
                
                # Update progress
                self.update_progress(
                    self.ui_components, 
                    percentage, 
                    100, 
                    f"{message} ({current}/{total})",
                    level
                )
            except Exception as e:
                self.logger.error(f"❌ Error updating progress: {str(e)}")
        
        return progress_callback
    
    def _extract_config(self) -> Dict[str, Any]:
        """Extract config dari UI components dengan proper error handling.
        
        Returns:
            Configuration dictionary
        """
        try:
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self.logger.warning("⚠️ Config handler tidak tersedia")
                return self._get_default_config()
                
            # Extract config dari UI
            if hasattr(config_handler, 'extract_config'):
                return config_handler.extract_config(self.ui_components)
            else:
                self.logger.warning("⚠️ Config handler tidak memiliki extract_config method")
                return self._get_default_config()
                
        except Exception as e:
            self.logger.error(f"❌ Error extracting config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk operation.
        
        Returns:
            Default configuration dictionary
        """
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            self.logger.error(f"❌ Error getting default config: {str(e)}")
            return {'data': {'dir': 'data'}, 'preprocessing': {'target_splits': ['train', 'valid']}}
