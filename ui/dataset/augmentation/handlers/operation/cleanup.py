"""
File: smartcash/ui/dataset/augmentation/handlers/operation/cleanup.py
Deskripsi: Cleanup operation handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
import logging
import ipywidgets as widgets

# Import base operation handler
from smartcash.ui.dataset.augmentation.handlers.operation.base_operation import BaseOperationHandler

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors


class CleanupOperationHandler(BaseOperationHandler):
    """Cleanup operation handler untuk augmentation module dengan centralized error handling
    
    Provides functionality for dataset cleanup operation:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize cleanup operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(ui_components=ui_components)
        self.logger.debug("CleanupOperationHandler initialized")
    
    @handle_ui_errors(log_error=True)
    def execute(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute cleanup operation dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Clear UI outputs using parent method
        self.clear_ui_outputs(self.ui_components)
        
        # Check if dialog is visible using parent method
        if self.is_confirmation_pending(self.ui_components):
            return {'status': False, 'message': 'Dialog sudah visible'}
        
        # Show confirmation dialog
        self._show_cleanup_confirmation_dialog()
        
        return {'status': True, 'message': 'Konfirmasi cleanup ditampilkan'}
    
    @handle_ui_errors(log_error=True)
    def execute_backend_operation(self) -> Dict[str, Any]:
        """Execute backend operation untuk cleanup
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Set button states
        self.disable_all_buttons(self.ui_components)
        
        # Create progress tracker
        if 'progress_tracker' in self.ui_components:
            self.ui_components['progress_tracker'].start("Memulai cleanup dataset...")
        
        # Execute backend operation
        result = self._execute_backend_operation('cleanup')
        
        # Handle result
        self._handle_cleanup_result(result)
        
        # Reset button states
        self.enable_all_buttons(self.ui_components)
        
        return result
    
    @handle_ui_errors(log_error=True)
    def _show_cleanup_confirmation_dialog(self) -> None:
        """Show cleanup confirmation dialog dengan centralized error handling"""
        # No need to import show_info_in_area as we're using parent class methods
        
        # Extract config
        config_handler = self.ui_components.get('config_handler')
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Get target directory
        target_dir = config.get('augmentation', {}).get('target_dir', '')
        
        # Create confirmation dialog
        confirm_button = widgets.Button(
            description='Confirm',
            button_style='success',
            icon='check'
        )
        
        cancel_button = widgets.Button(
            description='Cancel',
            button_style='danger',
            icon='times'
        )
        
        # Define button handlers
        def on_confirm_cleanup(btn):
            self.ui_components['_dialog_visible'] = False
            self.ui_components['_cleanup_confirmed'] = True
            self.execute_backend_operation()
        
        def on_cancel_cleanup(btn):
            self.ui_components['_dialog_visible'] = False
            self._handle_cleanup_cancel()
        
        # Bind button handlers
        confirm_button.on_click(on_confirm_cleanup)
        cancel_button.on_click(on_cancel_cleanup)
        
        # Show confirmation dialog
        self.ui_components['_dialog_visible'] = True
        show_info_in_area(
            self.ui_components,
            title="🗑️ Konfirmasi Cleanup Dataset",
            message=f"""
            <div style='background: #fff3cd; padding: 10px; border-radius: 4px; margin: 8px 0;'>
                <strong>⚠️ Warning:</strong><br>
                • Target Directory: {target_dir}<br>
                • Operasi ini akan menghapus file hasil augmentasi.<br>
                • File asli tidak akan terpengaruh.
            </div>
            <p>Apakah Anda yakin ingin melakukan cleanup dataset?</p>
            <p><strong>⚠️ Warning:</strong> Proses ini tidak dapat dibatalkan.</p>
            """,
            buttons=[confirm_button, cancel_button],
            dialog_type="warning"
        )
    
    @handle_ui_errors(log_error=True)
    def _execute_backend_operation(self, operation_type: str) -> Dict[str, Any]:
        """Execute backend operation dengan centralized error handling
        
        Args:
            operation_type: Tipe operasi backend
            
        Returns:
            Dictionary berisi hasil operasi
        """
        from smartcash.ui.dataset.augmentation.utils.backend_utils import execute_cleanup
        
        # Extract config
        config_handler = self.ui_components.get('config_handler')
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Define progress callback
        def progress_callback(level, current, total, message):
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker:
                if level == 'overall':
                    progress_tracker.update_overall(current / total * 100, message)
                elif level == 'step':
                    progress_tracker.update_step(current / total * 100, message)
                else:
                    progress_tracker.update(current, total, message)
            
            # Log progress using parent logger methods
            if level == "INFO":
                self.logger.info(message)
            elif level == "WARNING":
                self.logger.warning(message)
            elif level == "ERROR":
                self.logger.error(message)
            else:
                self.logger.debug(message)
        
        # Execute backend operation
        if operation_type == 'cleanup':
            self.logger.info("🧹 Memulai cleanup augmented files...")
            result = execute_cleanup(config)
        else:
            result = {'status': False, 'message': f'Operasi {operation_type} tidak didukung'}
        
        return result
    
    @handle_ui_errors(log_error=True)
    def _handle_cleanup_result(self, result: Dict[str, Any]) -> None:
        """Handle cleanup result dengan centralized error handling
        
        Args:
            result: Dictionary berisi hasil operasi
        """
        # Get progress tracker
        progress_tracker = self.ui_components.get('progress_tracker')
        
        # Check result status - using 'status' key for consistency (not 'success')
        if result.get('status'):
            # Get result details
            total_removed = result.get('total_removed', 0)
            target = result.get('target', 'both')
            target_split = result.get('target_split', 'train')
            
            # Log success message
            success_msg = f"✅ Cleanup berhasil: {total_removed} files dihapus (target: {target})"
            self.logger.info(success_msg)
            
            # Show success dialog using parent class method
            message = f"""
            <div style='background: #d4edda; padding: 10px; border-radius: 4px; margin: 8px 0;'>
                <strong>📊 Cleanup Summary:</strong><br>
                • Files Removed: {total_removed}<br>
                • Target: {target}<br>
                • Split: {target_split}
            </div>
            """
            message += """
                <p>Cleanup dataset telah selesai dengan sukses!</p>
                """
            self.show_info_dialog(
                ui_components=self.ui_components,
                title="✅ Cleanup Dataset Berhasil",
                message=message
            )
            
            # Update progress tracker
            if progress_tracker:
                progress_tracker.complete(success_msg)
            
            # Update summary panel
            self.update_operation_summary(result)
        else:
            # Get error message
            error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
            
            # Log error message
            self.log_error(error_msg)
            
            # Show error dialog using parent class method
            self.show_info_dialog(
                title="❌ Cleanup Dataset Gagal",
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
    
    @handle_ui_errors(log_error=True)
    def _handle_cleanup_cancel(self) -> None:
        """Handle cleanup cancellation dengan centralized error handling"""
        # Clear operation confirmed flag
        self.ui_components.pop('_cleanup_confirmed', None)
        
        # Log cancellation
        self.log_info("🚫 Cleanup dibatalkan oleh user")
    
    @handle_ui_errors(log_error=True)
    def set_button_states(self, disabled: bool = False) -> None:
        """Set button states dengan centralized error handling
        
        Args:
            disabled: True to disable buttons, False to enable
        """
        # Get buttons
        buttons = [
            'augment_button',
            'check_button',
            'cleanup_button',
            'preview_button',
            'save_button',
            'reset_button'
        ]
        
        # Set button states
        for button_key in buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = disabled
