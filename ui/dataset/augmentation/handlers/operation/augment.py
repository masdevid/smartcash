"""
File: smartcash/ui/dataset/augmentation/handlers/operation/augment.py
Deskripsi: Augment operation handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
import logging

# Import base operation handler
from smartcash.ui.dataset.augmentation.handlers.operation.base_operation import BaseOperationHandler

# Import error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors


class AugmentOperationHandler(BaseOperationHandler):
    """Augment operation handler untuk augmentation module dengan centralized error handling
    
    Provides functionality for augmentation operation:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize augment operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(ui_components=ui_components)
        self.logger.debug("AugmentOperationHandler initialized")
    
    @handle_ui_errors(log_error=True)
    def execute(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute augment operation dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Clear UI outputs using parent method
        self.clear_ui_outputs()
        
        # Check if dialog is visible using parent method
        if self.is_confirmation_pending(self.ui_components):
            return {'status': False, 'message': 'Dialog sudah visible'}
        
        # Validate form inputs
        validation = self._validate_form_inputs()
        if not validation['valid']:
            self._show_validation_errors(validation)
            return {'status': False, 'message': 'Validasi gagal'}
        
        # Show confirmation dialog
        self._show_augmentation_confirmation_dialog()
        
        return {'status': True, 'message': 'Konfirmasi augmentasi ditampilkan'}
    
    @handle_ui_errors(log_error=True)
    def _validate_form_inputs(self) -> Dict[str, Any]:
        """Validate form inputs dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil validasi
        """
        # Import validation functions from utils
        from smartcash.ui.dataset.augmentation.utils.validation_utils import validate_augmentation_form
        
        # Validate form inputs
        return validate_augmentation_form(self.ui_components)
    
    @handle_ui_errors(log_error=True)
    def execute_backend_operation(self) -> Dict[str, Any]:
        """Execute backend operation untuk augmentasi
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Set button states using parent method
        self.disable_all_buttons(self.ui_components)
        
        # Create progress tracker
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start("Memulai augmentasi dataset...")
        
        # Execute backend operation
        result = self._execute_backend_operation('augmentation_pipeline')
        
        # Handle result
        self._handle_pipeline_result(result)
        
        # Reset button states using parent method
        self.enable_all_buttons(self.ui_components)
        
        return result
    
    @handle_ui_errors(log_error=True)
    def _show_validation_errors(self, validation: Dict[str, Any]) -> None:
        """Show validation errors dengan centralized error handling
        
        Args:
            validation: Dictionary berisi hasil validasi
        """
        # Prepare error message
        error_list = "<ul>"
        for error in validation['errors']:
            error_list += f"<li>{error}</li>"
        error_list += "</ul>"
        
        # Create error message
        error_message = f"""
        <div style='background: #f8d7da; padding: 10px; border-radius: 4px; margin: 8px 0;'>
            <strong>Validasi gagal:</strong>
            {error_list}
        </div>
        <p>Silakan perbaiki input dan coba lagi.</p>
        """
        
        # Update status panel using parent method
        self.update_status_panel(
            self.ui_components,
            message=error_message,
            status_type="error",
            title="❌ Validasi Gagal"
        )
        
        # Log validation errors
        self.log_error(f"Validasi gagal: {len(validation['errors'])} errors")
    
    @handle_ui_errors(log_error=True)
    def _show_augmentation_confirmation_dialog(self) -> None:
        """Show augmentation confirmation dialog dengan centralized error handling"""
        from smartcash.ui.dataset.augmentation.utils.validation_utils import extract_augmentation_types
        
        # Extract augmentation types
        aug_types = extract_augmentation_types(self.ui_components)
        aug_types_str = ", ".join(aug_types) if aug_types else "None"
        
        # Get dataset path
        dataset_path = self.get_widget_value(self.ui_components, 'dataset_path_text', "/data/dataset")
        
        # Create confirmation message
        confirmation_message = f"""
        <div style='background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 8px 0;'>
            <strong>📋 Augmentation Configuration:</strong><br>
            • Dataset Path: <code>{dataset_path}</code><br>
            • Augmentation Types: <code>{aug_types_str}</code><br>
        </div>
        <p>Apakah Anda yakin ingin melakukan augmentasi dataset?</p>
        <p><small>⚠️ Proses ini akan membuat file baru di dataset path.</small></p>
        """
        
        # Show confirmation dialog using parent method
        self.show_confirmation_dialog(
            ui_components=self.ui_components,
            message=confirmation_message,
            callback=self._on_confirm_augmentation,
            title="🔄 Konfirmasi Augmentasi Dataset",
            confirm_text="Augmentasi",
            cancel_text="Batal"
        )
    
    def _on_confirm_augmentation(self) -> None:
        """Handle augmentation confirmation"""
        # Execute backend operation
        self.execute_backend_operation()
        
    def _on_cancel_augmentation(self) -> None:
        """Handle augmentation cancellation"""
        # Handle cancellation
        self._handle_augmentation_cancel()
    
    @handle_ui_errors(log_error=True)
    def _execute_backend_operation(self, operation_type: str) -> Dict[str, Any]:
        """Execute backend operation dengan centralized error handling
        
        Args:
            operation_type: Tipe operasi backend
            
        Returns:
            Dictionary berisi hasil operasi
        """
        from smartcash.ui.dataset.augmentation.utils.backend_utils import (
            execute_augmentation_pipeline,
            execute_dataset_check,
            execute_cleanup,
            create_live_preview
        )
        
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Extract config from UI
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Define progress callback
        def progress_callback(level, current, total, message):
            # Get progress tracker
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker:
                if level == 'overall':
                    progress_tracker.update_overall(current / total * 100, message)
                elif level == 'step':
                    progress_tracker.update_step(current / total * 100, message)
        
        # Execute backend operation
        if operation_type == 'augmentation_pipeline':
            self.log_info("🚀 Memulai augmentation pipeline...")
            result = execute_augmentation_pipeline(config, progress_callback)
        elif operation_type == 'dataset_check':
            self.log_info("🔍 Memulai comprehensive dataset check...")
            result = execute_dataset_check(config)
        elif operation_type == 'cleanup':
            self.log_info("🧹 Memulai cleanup augmented files...")
            result = execute_cleanup(config)
        elif operation_type == 'preview':
            self.log_info("📸 Memulai pembuatan preview...")
            result = create_live_preview(config)
        else:
            result = {'status': False, 'message': f'Operasi {operation_type} tidak didukung'}
        
        return result
    
    @handle_ui_errors(log_error=True)
    def _handle_pipeline_result(self, result: Dict[str, Any]) -> None:
        """Handle pipeline result dengan centralized error handling
        
        Args:
            result: Dictionary berisi hasil operasi
        """
        # Check if result is valid
        if not result:
            self.update_status_panel(
                self.ui_components,
                message="<p>Terjadi kesalahan saat menjalankan augmentasi.</p>",
                status_type="error",
                title="❌ Error"
            )
            return
        
        # Check if operation was successful - use 'status' key for API consistency
        if result.get('status', False):
            # Get augmented files
            augmented_files = result.get('augmented_files', [])
            augmented_count = len(augmented_files)
            
            # Create success message
            success_message = f"""
            <div style='background: #d4edda; padding: 10px; border-radius: 4px; margin: 8px 0;'>
                <strong>✅ Augmentasi selesai:</strong><br>
                • {augmented_count} file berhasil diaugmentasi<br>
                • Output disimpan di: <code>{result.get('output_dir', 'N/A')}</code>
            </div>
            <p>Augmentasi dataset berhasil dilakukan.</p>
            """
            
            # Update summary container
            self.update_summary_container(
                self.ui_components,
                message=success_message,
                title="✅ Augmentasi Berhasil"
            )
            
            # Log success
            self.log_info(f"Augmentasi berhasil: {augmented_count} file diaugmentasi")
        else:
            # Create error message
            error_message = f"""
            <div style='background: #f8d7da; padding: 10px; border-radius: 4px; margin: 8px 0;'>
                <strong>❌ Error:</strong><br>
                {result.get('message', 'Unknown error')}
            </div>
            <p>Augmentasi dataset gagal. Silakan coba lagi.</p>
            """
            
            # Update status panel
            self.update_status_panel(
                self.ui_components,
                message=error_message,
                status_type="error",
                title="❌ Augmentasi Gagal"
            )
            
            # Log error
            self.log_error(f"Augmentasi gagal: {result.get('message', 'Unknown error')}")
    
    @handle_ui_errors(log_error=True)
    def _handle_augmentation_cancel(self) -> None:
        """Handle augmentation cancellation dengan centralized error handling"""
        # Update status panel
        self.update_status_panel(
            self.ui_components,
            message="<p>Augmentasi dataset dibatalkan oleh pengguna.</p>",
            status_type="warning",
            title="⚠️ Augmentasi Dibatalkan"
        )
            
        # Log warning
        self.log_warning("Augmentasi dibatalkan oleh pengguna")
    
# Duplicate method removed
