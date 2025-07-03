"""
File: smartcash/ui/dataset/augmentation/handlers/operation/check.py
Deskripsi: Check operation handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
import logging

# Import base operation handler
from smartcash.ui.dataset.augmentation.handlers.operation.base_operation import BaseOperationHandler

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors


class CheckOperationHandler(BaseOperationHandler):
    """Check operation handler untuk augmentation module dengan centralized error handling
    
    Provides functionality for dataset check operation:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize check operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(ui_components=ui_components)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("CheckOperationHandler initialized")
    
    @handle_ui_errors(log_error=True)
    def execute(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute check operation dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Clear UI outputs using parent class method
        self.clear_ui_outputs()
        
        # Check if dialog is visible using parent class method
        if self.is_confirmation_pending(self.ui_components):
            return {'status': False, 'message': 'Dialog sudah visible'}
        
        # Execute backend operation
        result = self._execute_backend_operation('dataset_check')
        
        # Handle result
        self._handle_check_result(result)
        
        return result
    
    @handle_ui_errors(log_error=True)
    def _execute_backend_operation(self, operation_type: str) -> Dict[str, Any]:
        """Execute backend operation dengan centralized error handling
        
        Args:
            operation_type: Tipe operasi backend
            
        Returns:
            Dictionary berisi hasil operasi
        """
        from smartcash.ui.dataset.augmentation.utils.backend_utils import execute_dataset_check
        
        # Set button states
        self.disable_all_buttons(self.ui_components)
        
        # Create progress tracker
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start("Memeriksa dataset...")
        
        # Extract config
        config_handler = self.ui_components.get('config_handler')
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Define progress callback
        def progress_callback(level, current, total, message):
            if progress_tracker and hasattr(progress_tracker, 'update'):
                if level == 'overall':
                    progress_tracker.update_overall(current / total * 100, message)
                elif level == 'step':
                    progress_tracker.update_step(current / total * 100, message)
                else:
                    progress_tracker.update(current, total, message)
            
            # Log progress using parent class methods
            if level == "INFO":
                self.log_info(message)
            elif level == "WARNING":
                self.log_warning(message)
            elif level == "ERROR":
                self.log_error(message)
            else:
                self.log_debug(message)
        
        # Execute backend operation
        if operation_type == 'dataset_check':
            self.log_info("🔍 Memulai comprehensive dataset check...")
            result = execute_dataset_check(config)
        else:
            result = {'status': False, 'message': f'Operasi {operation_type} tidak didukung'}
        
        # Reset button states
        self.enable_all_buttons(self.ui_components)
        
        return result
    
    @handle_ui_errors(log_error=True)
    def _handle_check_result(self, result: Dict[str, Any]) -> None:
        """Handle check result dengan centralized error handling
        
        Args:
            result: Dictionary berisi hasil operasi
        """
        # No need to import show_info_in_area as we're using parent class methods
        
        # Get progress tracker
        progress_tracker = self.ui_components.get('progress_tracker')
        
        # Check result status - using 'status' key for consistency (not 'success')
        if result.get('status'):
            # Get result details
            total_files = result.get('total_files', 0)
            valid_files = result.get('valid_files', 0)
            invalid_files = result.get('invalid_files', 0)
            
            # Log success message using parent class method
            success_msg = f"✅ Dataset check berhasil: {valid_files}/{total_files} files valid"
            self.log_info(success_msg)
            
            # Show success dialog using parent class method
            self.show_info_dialog(
                title="✅ Dataset Check Berhasil",
                message=f"""
                <div style='background: #d4edda; padding: 10px; border-radius: 4px; margin: 8px 0;'>
                    <strong>📊 Dataset Check Summary:</strong><br>
                    • Total Files: {total_files}<br>
                    • Valid Files: {valid_files}<br>
                    • Invalid Files: {invalid_files}<br>
                </div>
                <p>Dataset check telah selesai dengan sukses!</p>
                """,
                dialog_type="success"
            )
            
            # Update progress tracker
            if progress_tracker:
                progress_tracker.complete(success_msg)
            
            # Update summary panel
            self.update_operation_summary(result)
        else:
            # Get error message
            error_msg = f"❌ Dataset check gagal: {result.get('message', 'Unknown error')}"
            
            # Log error message using parent class method
            self.log_error(error_msg)
            
            # Show error dialog using parent class method
            self.show_info_dialog(
                title="❌ Dataset Check Gagal",
                message=f"""
                <div style='background: #f8d7da; padding: 10px; border-radius: 4px; margin: 8px 0;'>
                    <strong>Error:</strong> {result.get('message', 'Unknown error')}
                </div>
                <p>Check file permissions atau coba lagi.</p>
                """,
                dialog_type="error"
            )
            
            # Update progress tracker
            if progress_tracker:
                progress_tracker.error(error_msg)
            
            # Update summary panel
            self.update_operation_summary(result)
    
    # Removed redundant set_button_states method - using parent class methods instead
    # Parent class methods to use:
    # - self.set_buttons_state(self.ui_components, not disabled)
    # - self.enable_all_buttons(self.ui_components)
    # - self.disable_all_buttons(self.ui_components)
